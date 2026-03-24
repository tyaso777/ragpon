from __future__ import annotations

import base64
import hashlib
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

# -----------------------------
# Configurable constants
# -----------------------------
MAIN_BODY_ITEM_NAMES: set[str] = {"body", "body_1", "rtf", "richtext"}
ATTACHMENT_BODY_ITEM_NAMES: set[str] = {"tmp_i", "tmp", "temp_i", "temp"}

ATTACHMENTS_SECTION_PREFIX: str = "### 添付資料"

SKIP_CATEGORY_VALUES: set[str] = {"一", "1"}

# Inline base64 media tags; decode and save only if standard image
INLINE_BASE64_MEDIA_TAGS: set[str] = {"notesbitmap", "gif"}

_BASE64_CHARS_RE = re.compile(r"^[A-Za-z0-9+/=]+$")
_INVALID_FILENAME_RE = re.compile(r'[<>:"/\\|?*]+')


RenderContext = Literal["flow", "table_cell"]


@dataclass
class NotesImage:
    """Represents an extracted image asset."""

    image_id: str
    rel_path: str
    content_hash: str
    mime: str | None
    source_tag: str
    size_bytes: int


@dataclass
class NotesAttachment:
    """Represents an extracted attachment file."""

    attachment_id: str
    filename: str
    rel_path: str
    sha256: str
    size_bytes: int
    declared_size: int | None
    created: str | None
    modified: str | None
    hosttype: str | None
    flags: str | None
    compression: str | None


@dataclass
class NotesMetadata:
    """Represents metadata for a Notes document."""

    database_title: str | None
    replica_id: str | None
    unid: str | None
    noteid: str | None
    form: str | None
    subject: str | None
    http_link: str | None
    notes_link: str | None
    owner_department: str | None
    categories: list[str]


@dataclass
class NotesDocument:
    """Represents a normalized Notes document for RAG indexing."""

    doc_id: str
    metadata: NotesMetadata
    body_text: str
    images: list[NotesImage]
    attachments: list[NotesAttachment]


# -----------------------------
# Utilities
# -----------------------------
def _strip_ns(tag: str) -> str:
    """Strip XML namespace from tag."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _normalize_item_name(name_raw: str) -> str:
    """Normalize <item name="..."> key only (not body text)."""
    return unicodedata.normalize("NFKC", name_raw).strip().lower()


def _safe_dirname(value: str) -> str:
    """Make a safe directory name from doc_id-like strings."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _sanitize_filename(name: str) -> str:
    """Sanitize filename for cross-platform compatibility."""
    sanitized = _INVALID_FILENAME_RE.sub("_", name)
    sanitized = sanitized.rstrip(" .")
    return sanitized if sanitized else "attachment"


def _get_note_identifiers(note_elem: ET.Element) -> tuple[str | None, str | None]:
    """Extract UNID and NoteID from a <note> or <document> element."""
    unid: str | None = None
    noteid: str | None = None

    noteinfo = note_elem.find(".//{*}noteinfo")
    if noteinfo is not None:
        unid = noteinfo.attrib.get("unid") or unid
        noteid = noteinfo.attrib.get("noteid") or noteid

        if unid is None:
            unid_elem = noteinfo.find(".//{*}unid")
            if unid_elem is not None and unid_elem.text:
                unid = unid_elem.text.strip()

        if noteid is None:
            noteid_elem = noteinfo.find(".//{*}noteid")
            if noteid_elem is not None and noteid_elem.text:
                noteid = noteid_elem.text.strip()

    if unid is None:
        unid = note_elem.attrib.get("unid")
    if noteid is None:
        noteid = note_elem.attrib.get("noteid") or note_elem.attrib.get("noteID")

    return unid, noteid


def _extract_item_text(item_elem: ET.Element) -> str | None:
    """Extract a simple <text> value from an <item> element."""
    text_elem = item_elem.find(".//{*}text")
    if text_elem is not None and text_elem.text:
        return text_elem.text.strip()
    return None


def _normalize_optional_value(value: str | None) -> str | None:
    """Normalize optional string values."""
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed if trimmed else None


def _build_notes_link(
    base: str | None, replica_id: str | None, unid: str | None
) -> str | None:
    """Build a Notes link string from base URL, replica ID, and UNID."""
    if not base or not replica_id or not unid:
        return None
    return f"{base}{replica_id}/0/{unid}"


def _escape_markdown_link_text(text: str) -> str:
    """Escape characters that can break Markdown link text."""
    return text.replace("[", "\\[").replace("]", "\\]")


def _escape_markdown_url(url: str) -> str:
    """Escape characters that can break Markdown link URL."""
    return url.replace("(", "%28").replace(")", "%29")


# -----------------------------
# Base64 decoding helpers
# -----------------------------
def _compact_base64(raw: str) -> str | None:
    """Remove whitespace and fix padding for base64."""
    compact = re.sub(r"\s+", "", raw.strip())
    if len(compact) < 64:
        return None
    if not _BASE64_CHARS_RE.match(compact):
        return None

    pad_len = (-len(compact)) % 4
    if pad_len:
        compact += "=" * pad_len
    return compact


def _decode_base64_bytes(b64: str) -> bytes | None:
    """Decode base64 to bytes. Returns None if decoding fails."""
    try:
        return base64.b64decode(b64, validate=True)
    except Exception:
        try:
            return base64.b64decode(b64, validate=False)
        except Exception:
            return None


# -----------------------------
# Image decoding
# -----------------------------
def _detect_image_type(data: bytes) -> tuple[str | None, str | None]:
    """Detect image type by magic bytes."""
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif", "image/gif"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png", "image/png"
    if data.startswith(b"\xff\xd8"):
        return "jpg", "image/jpeg"
    if data.startswith(b"BM"):
        return "bmp", "image/bmp"
    if data.startswith(b"II*\x00") or data.startswith(b"MM\x00*"):
        return "tif", "image/tiff"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    return None, None


def _extract_base64_from_tag_text(elem: ET.Element) -> str | None:
    """Extract base64 text from a media tag using elem.text only."""
    if elem.text is None:
        return None
    return _compact_base64(elem.text)


def _save_image_if_standard(
    *,
    b64: str,
    source_tag: str,
    doc_images_dir: Path,
    doc_images_rel_prefix: str,
    image_id: str,
) -> NotesImage | None:
    """Decode base64 and save only if it is a standard image."""
    data = _decode_base64_bytes(b64)
    if not data:
        return None

    ext, mime = _detect_image_type(data)
    if ext is None:
        return None

    doc_images_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{image_id}.{ext}"
    (doc_images_dir / filename).write_bytes(data)

    sha256_hex = hashlib.sha256(data).hexdigest()
    rel_path = f"{doc_images_rel_prefix}{filename}"

    return NotesImage(
        image_id=image_id,
        rel_path=rel_path,
        content_hash=sha256_hex,
        mime=mime,
        source_tag=source_tag,
        size_bytes=len(data),
    )


# -----------------------------
# Attachment decoding ($FILE)
# -----------------------------
def _extract_datetime_text(parent: ET.Element, tag: str) -> str | None:
    """Extract timestamps like <created><datetime>...</datetime></created>."""
    elem = parent.find(f".//{{*}}{tag}//{{*}}datetime")
    if elem is not None and elem.text:
        return elem.text.strip()
    return None


def _extract_int_attr(elem: ET.Element, attr: str) -> int | None:
    """Parse integer attribute safely."""
    raw = elem.attrib.get(attr)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _save_attachment_from_file_elem(
    *,
    file_elem: ET.Element,
    attachments_dir: Path,
    attachments_rel_prefix: str,
    attachment_id: str,
) -> NotesAttachment | None:
    """Decode <filedata> from a <file> element and save attachment."""
    original_name = file_elem.attrib.get("name") or "attachment"
    safe_name = _sanitize_filename(original_name)

    filedata_elem = file_elem.find(".//{*}filedata")
    if filedata_elem is None or filedata_elem.text is None:
        return None

    compact = _compact_base64(filedata_elem.text)
    if compact is None:
        return None

    data = _decode_base64_bytes(compact)
    if not data:
        return None

    attachments_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"{attachment_id}_{safe_name}"
    out_path = attachments_dir / out_filename
    out_path.write_bytes(data)

    sha256_hex = hashlib.sha256(data).hexdigest()
    declared_size = _extract_int_attr(file_elem, "size")
    created = _extract_datetime_text(file_elem, "created")
    modified = _extract_datetime_text(file_elem, "modified")

    return NotesAttachment(
        attachment_id=attachment_id,
        filename=original_name,
        rel_path=f"{attachments_rel_prefix}{out_filename}",
        sha256=sha256_hex,
        size_bytes=len(data),
        declared_size=declared_size,
        created=created,
        modified=modified,
        hosttype=file_elem.attrib.get("hosttype"),
        flags=file_elem.attrib.get("flags"),
        compression=file_elem.attrib.get("compression"),
    )


def _extract_attachments_from_file_item(
    *,
    file_item_elem: ET.Element,
    doc_attachments_dir: Path,
    doc_attachments_rel_prefix: str,
) -> tuple[list[NotesAttachment], list[str]]:
    """Extract all attachments from an <item name="$FILE"> element."""
    attachments: list[NotesAttachment] = []
    md_lines: list[str] = []

    file_elems = file_item_elem.findall(".//{*}file")
    counter = 0

    for file_elem in file_elems:
        counter += 1
        attachment_id = f"att_{counter:04d}"

        saved = _save_attachment_from_file_elem(
            file_elem=file_elem,
            attachments_dir=doc_attachments_dir,
            attachments_rel_prefix=doc_attachments_rel_prefix,
            attachment_id=attachment_id,
        )
        if saved is None:
            continue

        attachments.append(saved)
        label = _escape_markdown_link_text(saved.filename)
        url = _escape_markdown_url(saved.rel_path)
        md_lines.append(f"- [{label}]({url})")

    return attachments, md_lines


# -----------------------------
# Markdown rendering (context-aware caption)
# -----------------------------
def _render_node_to_markdown(
    elem: ET.Element,
    http_base: str | None,
    notes_base: str | None,
    prefer_http: bool,
    *,
    context: RenderContext,
    attachment_list_mode: bool,
    doc_images_dir: Path | None,
    doc_images_rel_prefix: str | None,
    images_out: list[NotesImage],
    image_counter: list[int],
) -> str:
    """Render an XML element subtree into Markdown."""
    tag_name = _strip_ns(elem.tag) if isinstance(elem.tag, str) else ""

    # Context-aware caption rendering
    if tag_name == "caption":
        caption_text = "".join(elem.itertext()).strip()
        if not caption_text:
            return ""
        if context == "flow" and attachment_list_mode:
            return f"- {caption_text}\n"
        return f"{caption_text}\n"

    # Inline base64 media tags: decode once, save only if standard image
    if tag_name in INLINE_BASE64_MEDIA_TAGS:
        if doc_images_dir is None or doc_images_rel_prefix is None:
            return ""

        b64 = _extract_base64_from_tag_text(elem)
        if not b64:
            return ""

        image_counter[0] += 1
        image_id = f"img_{image_counter[0]:04d}"

        saved = _save_image_if_standard(
            b64=b64,
            source_tag=tag_name,
            doc_images_dir=doc_images_dir,
            doc_images_rel_prefix=doc_images_rel_prefix,
            image_id=image_id,
        )
        if saved is None:
            return ""

        images_out.append(saved)
        return f"![]({saved.rel_path})"

    if tag_name == "urllink":
        href = elem.attrib.get("href") or ""
        inner_parts: list[str] = []

        if elem.text:
            inner_parts.append(elem.text)

        for child in elem:
            inner_parts.append(
                _render_node_to_markdown(
                    child,
                    http_base,
                    notes_base,
                    prefer_http,
                    context=context,
                    attachment_list_mode=attachment_list_mode,
                    doc_images_dir=doc_images_dir,
                    doc_images_rel_prefix=doc_images_rel_prefix,
                    images_out=images_out,
                    image_counter=image_counter,
                )
            )
            if child.tail:
                inner_parts.append(child.tail)

        inner_text = "".join(inner_parts).strip() or href
        safe_text = _escape_markdown_link_text(inner_text)
        safe_href = _escape_markdown_url(href)
        return f"[{safe_text}]({safe_href})" if safe_href else safe_text

    if tag_name == "doclink":
        replica_id = elem.attrib.get("database")
        unid = elem.attrib.get("document")
        desc = (elem.attrib.get("description") or "").strip()

        label = desc if desc else "添付資料"
        safe_label = _escape_markdown_link_text(label)

        target_base = http_base if prefer_http else notes_base
        url = _build_notes_link(target_base, replica_id, unid)
        if url:
            return f"[{safe_label}]({_escape_markdown_url(url)})"
        return safe_label

    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)

    for child in elem:
        parts.append(
            _render_node_to_markdown(
                child,
                http_base,
                notes_base,
                prefer_http,
                context=context,
                attachment_list_mode=attachment_list_mode,
                doc_images_dir=doc_images_dir,
                doc_images_rel_prefix=doc_images_rel_prefix,
                images_out=images_out,
                image_counter=image_counter,
            )
        )
        if child.tail:
            parts.append(child.tail)

    return "".join(parts)


def _render_paragraph_to_markdown(
    par_elem: ET.Element,
    http_base: str | None,
    notes_base: str | None,
    prefer_http: bool,
    *,
    attachment_list_mode: bool,
    doc_images_dir: Path | None,
    doc_images_rel_prefix: str | None,
    images_out: list[NotesImage],
    image_counter: list[int],
) -> str:
    """Render a <par> element into Markdown."""
    return _render_node_to_markdown(
        par_elem,
        http_base,
        notes_base,
        prefer_http,
        context="flow",
        attachment_list_mode=attachment_list_mode,
        doc_images_dir=doc_images_dir,
        doc_images_rel_prefix=doc_images_rel_prefix,
        images_out=images_out,
        image_counter=image_counter,
    ).strip()


def _ensure_grid_size(
    grid: list[list[str | None]], row_index: int, col_index: int
) -> None:
    """Ensure grid has at least row_index+1 rows and col_index+1 columns."""
    while len(grid) <= row_index:
        grid.append([])
    row = grid[row_index]
    if len(row) <= col_index:
        row.extend([None] * (col_index + 1 - len(row)))


def _grid_to_markdown(grid: list[list[str | None]]) -> str:
    """Convert a 2D grid into a Markdown table string."""
    if not grid:
        return ""

    max_cols = max(len(row) for row in grid)
    normalized: list[list[str]] = []
    for row in grid:
        normalized.append(
            [
                row[c] if c < len(row) and row[c] is not None else ""
                for c in range(max_cols)
            ]
        )

    lines: list[str] = []
    header = normalized[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")

    for row in normalized[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _normalize_table_cell_text(text: str) -> str:
    """Normalize table cell text for Markdown table compatibility."""
    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    stripped = normalized_newlines.strip()
    replaced_newlines = stripped.replace("\n", "<br>")
    return replaced_newlines.replace("|", "\\|")


def _parse_table_to_markdown(
    table_elem: ET.Element,
    http_base: str | None,
    notes_base: str | None,
    prefer_http: bool,
    *,
    attachment_list_mode: bool,
    doc_images_dir: Path | None,
    doc_images_rel_prefix: str | None,
    images_out: list[NotesImage],
    image_counter: list[int],
) -> str:
    """Parse a <table> element and return Markdown."""
    grid: list[list[str | None]] = []
    row_tags = {"tablerow", "row"}
    cell_tags = {"tablecell", "cell"}

    row_elems = [child for child in table_elem if _strip_ns(child.tag) in row_tags]

    for r_physical, row_elem in enumerate(row_elems):
        _ensure_grid_size(grid, r_physical, 0)
        cell_elems = [child for child in row_elem if _strip_ns(child.tag) in cell_tags]

        col_index = 0
        for cell_elem in cell_elems:
            while True:
                _ensure_grid_size(grid, r_physical, col_index)
                if grid[r_physical][col_index] is None:
                    break
                col_index += 1

            raw_cell_md = _render_node_to_markdown(
                cell_elem,
                http_base,
                notes_base,
                prefer_http,
                context="table_cell",
                attachment_list_mode=attachment_list_mode,
                doc_images_dir=doc_images_dir,
                doc_images_rel_prefix=doc_images_rel_prefix,
                images_out=images_out,
                image_counter=image_counter,
            )
            text = _normalize_table_cell_text(raw_cell_md)

            raw_rowspan = cell_elem.attrib.get("rowspan") or "1"
            raw_colspan = (
                cell_elem.attrib.get("colspan")
                or cell_elem.attrib.get("columnspan")
                or "1"
            )

            try:
                rowspan = max(int(raw_rowspan), 1)
            except ValueError:
                rowspan = 1

            try:
                colspan = max(int(raw_colspan), 1)
            except ValueError:
                colspan = 1

            for dr in range(rowspan):
                r_target = r_physical + dr
                for dc in range(colspan):
                    c_target = col_index + dc
                    _ensure_grid_size(grid, r_target, c_target)
                    if grid[r_target][c_target] is None:
                        grid[r_target][c_target] = text

            col_index += colspan

    return _grid_to_markdown(grid)


def _extract_body_blocks_as_markdown(
    item_elem: ET.Element,
    http_base: str | None,
    notes_base: str | None,
    prefer_http: bool,
    *,
    attachment_list_mode: bool,
    doc_images_dir: Path | None,
    doc_images_rel_prefix: str | None,
    images_out: list[NotesImage],
    image_counter: list[int],
) -> list[str]:
    """Extract body as a list of markdown blocks (paragraphs and tables)."""
    blocks: list[str] = []

    richtext = item_elem.find(".//{*}richtext")
    if richtext is None:
        for par_elem in item_elem.findall(".//{*}par"):
            text = _render_paragraph_to_markdown(
                par_elem,
                http_base,
                notes_base,
                prefer_http,
                attachment_list_mode=attachment_list_mode,
                doc_images_dir=doc_images_dir,
                doc_images_rel_prefix=doc_images_rel_prefix,
                images_out=images_out,
                image_counter=image_counter,
            )
            if text:
                blocks.append(text)
        return blocks

    for child in richtext:
        tag_name = _strip_ns(child.tag)

        if tag_name == "par":
            text = _render_paragraph_to_markdown(
                child,
                http_base,
                notes_base,
                prefer_http,
                attachment_list_mode=attachment_list_mode,
                doc_images_dir=doc_images_dir,
                doc_images_rel_prefix=doc_images_rel_prefix,
                images_out=images_out,
                image_counter=image_counter,
            )
            if text:
                blocks.append(text)

        elif tag_name == "table":
            md = _parse_table_to_markdown(
                child,
                http_base,
                notes_base,
                prefer_http,
                attachment_list_mode=attachment_list_mode,
                doc_images_dir=doc_images_dir,
                doc_images_rel_prefix=doc_images_rel_prefix,
                images_out=images_out,
                image_counter=image_counter,
            )
            if md:
                blocks.append(md)

    return blocks


# -----------------------------
# Main parsing
# -----------------------------
def _parse_note_element(
    note_elem: ET.Element,
    default_replica_id: str | None,
    database_title: str | None,
    http_base: str | None,
    notes_base: str | None,
    prefer_http: bool,
    *,
    output_dir: Path,
) -> NotesDocument:
    """Parse a single <note> or <document> element into a NotesDocument."""
    replica_id = note_elem.attrib.get("replicaid") or default_replica_id
    unid, noteid = _get_note_identifiers(note_elem)

    form: str | None = note_elem.attrib.get("form")
    subject: str | None = None
    owner_department: str | None = None
    categories_map: dict[str, str | None] = {f"cate{i}": None for i in range(1, 6)}

    provisional_doc_id = (
        f"{replica_id}:{unid}" if replica_id and unid else (unid or noteid or "unknown")
    )
    doc_dirname = _safe_dirname(provisional_doc_id)

    # Image output setup
    doc_images_dir = output_dir / "images" / doc_dirname
    doc_images_rel_prefix = f"images/{doc_dirname}/"
    images_out: list[NotesImage] = []
    image_counter = [0]

    # Attachment output setup
    doc_attachments_dir = output_dir / "attachments" / doc_dirname
    doc_attachments_rel_prefix = f"attachments/{doc_dirname}/"
    attachments_out: list[NotesAttachment] = []
    attachments_md_lines: list[str] = []

    main_blocks: list[str] = []
    attachment_blocks: list[str] = []

    for item_elem in note_elem.findall(".//*"):
        if _strip_ns(item_elem.tag) != "item":
            continue

        name_lower = _normalize_item_name(item_elem.attrib.get("name", ""))

        if name_lower in {"subject", "title"}:
            subject = (
                _normalize_optional_value(_extract_item_text(item_elem)) or subject
            )

        if name_lower == "otherdepnm":
            owner_department = _normalize_optional_value(_extract_item_text(item_elem))

        if name_lower in {f"cate{i}" for i in range(1, 6)}:
            categories_map[name_lower] = _normalize_optional_value(
                _extract_item_text(item_elem)
            )

        if form is None and name_lower == "form":
            form = _normalize_optional_value(_extract_item_text(item_elem)) or form

        if name_lower in MAIN_BODY_ITEM_NAMES:
            main_blocks.extend(
                _extract_body_blocks_as_markdown(
                    item_elem,
                    http_base,
                    notes_base,
                    prefer_http,
                    attachment_list_mode=False,
                    doc_images_dir=doc_images_dir,
                    doc_images_rel_prefix=doc_images_rel_prefix,
                    images_out=images_out,
                    image_counter=image_counter,
                )
            )

        if name_lower in ATTACHMENT_BODY_ITEM_NAMES:
            attachment_blocks.extend(
                _extract_body_blocks_as_markdown(
                    item_elem,
                    http_base,
                    notes_base,
                    prefer_http,
                    attachment_list_mode=True,
                    doc_images_dir=doc_images_dir,
                    doc_images_rel_prefix=doc_images_rel_prefix,
                    images_out=images_out,
                    image_counter=image_counter,
                )
            )

        if name_lower == "$file":
            atts, md_lines = _extract_attachments_from_file_item(
                file_item_elem=item_elem,
                doc_attachments_dir=doc_attachments_dir,
                doc_attachments_rel_prefix=doc_attachments_rel_prefix,
            )
            attachments_out.extend(atts)
            attachments_md_lines.extend(md_lines)

    # Final doc_id
    if replica_id and unid:
        doc_id = f"{replica_id}:{unid}"
    elif unid:
        doc_id = unid
    else:
        doc_id = noteid or "unknown"

    http_link = _build_notes_link(http_base, replica_id, unid)
    notes_link = _build_notes_link(notes_base, replica_id, unid)

    categories: list[str] = []
    for i in range(1, 6):
        value = categories_map.get(f"cate{i}")
        if not value:
            continue
        if value in SKIP_CATEGORY_VALUES:
            continue
        categories.append(value)

    metadata = NotesMetadata(
        database_title=database_title,
        replica_id=replica_id,
        unid=unid,
        noteid=noteid,
        form=form,
        subject=subject,
        http_link=http_link,
        notes_link=notes_link,
        owner_department=owner_department,
        categories=categories,
    )

    merged_blocks: list[str] = [b for b in main_blocks if b]

    filtered_attachment_blocks = [b for b in attachment_blocks if b]
    has_attachments_section = bool(filtered_attachment_blocks or attachments_md_lines)

    if has_attachments_section:
        merged_blocks.append(ATTACHMENTS_SECTION_PREFIX)

        # Tmp_I / Temp body content (may contain tables; preserved)
        merged_blocks.extend(filtered_attachment_blocks)

        # $FILE attachments list is explicitly a list block
        if attachments_md_lines:
            merged_blocks.append("#### ファイル")
            merged_blocks.append("\n".join(attachments_md_lines))

    body_text = "\n\n".join(merged_blocks)

    return NotesDocument(
        doc_id=doc_id,
        metadata=metadata,
        body_text=body_text,
        images=images_out,
        attachments=attachments_out,
    )


def parse_dxl_file(
    path: Path,
    *,
    http_base: str | None = None,
    notes_base: str | None = None,
    prefer_http: bool = False,
    output_dir: Path | None = None,
) -> list[NotesDocument]:
    """Parse a DXL (XML) file and extract Notes documents."""
    if not path.exists():
        raise FileNotFoundError(f"DXL file not found: {path}")

    if output_dir is None:
        output_dir = Path.cwd()

    tree = ET.parse(path)
    root = tree.getroot()
    replica_id = root.attrib.get("replicaid") or root.attrib.get("replicaID")
    database_title = root.attrib.get("title")

    docs: list[NotesDocument] = []
    for note_elem in root.iter():
        if _strip_ns(note_elem.tag) not in {"note", "document"}:
            continue
        docs.append(
            _parse_note_element(
                note_elem=note_elem,
                default_replica_id=replica_id,
                database_title=database_title,
                http_base=http_base,
                notes_base=notes_base,
                prefer_http=prefer_http,
                output_dir=output_dir,
            )
        )
    return docs


def parse_dxl_string(
    xml_text: str,
    *,
    http_base: str | None = None,
    notes_base: str | None = None,
    prefer_http: bool = False,
    output_dir: Path | None = None,
) -> list[NotesDocument]:
    """Parse a DXL XML string and extract Notes documents."""
    if output_dir is None:
        output_dir = Path.cwd()

    root = ET.fromstring(xml_text)
    replica_id = root.attrib.get("replicaid") or root.attrib.get("replicaID")
    database_title = root.attrib.get("title")

    docs: list[NotesDocument] = []
    for note_elem in root.iter():
        if _strip_ns(note_elem.tag) not in {"note", "document"}:
            continue
        docs.append(
            _parse_note_element(
                note_elem=note_elem,
                default_replica_id=replica_id,
                database_title=database_title,
                http_base=http_base,
                notes_base=notes_base,
                prefer_http=prefer_http,
                output_dir=output_dir,
            )
        )
    return docs


def export_to_ndjson(docs: list[NotesDocument], output_path: Path) -> None:
    """Export NotesDocument objects to a NDJSON file."""
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for doc in docs:
        data: dict[str, Any] = {
            "doc_id": doc.doc_id,
            "metadata": asdict(doc.metadata),
            "body_text": doc.body_text,
            "images": [asdict(img) for img in doc.images],
            "attachments": [asdict(att) for att in doc.attachments],
        }
        lines.append(json.dumps(data, ensure_ascii=False))

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    HTTP_BASE = "https://example.com/db/"
    NOTES_BASE = "notes://server/db/"
    PREFER_HTTP = False

    dxl_path = Path("input.dxl")
    out_path = Path("notes_docs.ndjson")

    docs = parse_dxl_file(
        dxl_path,
        http_base=HTTP_BASE,
        notes_base=NOTES_BASE,
        prefer_http=PREFER_HTTP,
        output_dir=out_path.parent,
    )
    export_to_ndjson(docs, out_path)
    print(f"Exported {len(docs)} documents to {out_path}")
