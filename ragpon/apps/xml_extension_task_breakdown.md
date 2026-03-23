# XML由来JSON取り込み機能拡張 タスク分解メモ

## 目的

- 既存の CSV 前提の取り込みフローに加えて、XML から加工済みの JSON データも ChromaDB に格納できるようにする。
- XML のメタデータ `notes_link` を、Streamlit の「回答に使用された情報を見る」に表示できるようにする。
- `images` と `attachments` は今回のスコープ外とする。

## 現状整理

- `DocumentProcessingService.process_file()` は拡張子に応じた `DocumentReader` でファイルを読み、`FilePathDocumentProcessingPipeline` で chunk と metadata を生成している。
- `DocumentProcessingService.process_dataframe()` は DataFrame の 1 行を 1 ドキュメントとして metadata ごと保存できる。
- 現状の `process_dataframe()` は `process_file()` と違って本文を chunk 分割していない。
- ChromaDB には `BaseDocument.metadata` の中身だけが保存される。
- FastAPI は検索結果から `doc_id`、`semantic distance`、`Text` を system message 用の文字列に変換している。
- Streamlit はその文字列を再パースして、「回答に使用された情報を見る」を表示している。

## 重要な前提

- 取り込み対象は XML そのものではなく、XML から加工済みの JSON ファイルである。
- サンプルの `xml_example.json` は、XML 1 record 相当のデータが並ぶ JSON Lines 形式に見える。
- 業務要件としては「元 XML の各 record を、JSON 上でも 1 つの CSV 相当として扱う」が本質。

## タスク一覧

### 1. JSON 取り込み仕様を確定する

- 加工済み JSON の入力形式を決める。
- 1 record から抽出する対象フィールドを決める。
- chunk 対象テキストを `body_text` のみにするか、`subject` などを前置するかを決める。
- `images` と `attachments` を完全に無視する方針を明文化する。

### 1.1 JSON 正規化後の列定義案

- JSON 1 record を、まず以下の列を持つ DataFrame 1 row に正規化する。
- `source_file_path` と `source_file_modified_at` も DataFrame 列として持たせる方針にする。
- 第一候補:
  - `source_doc_id`
  - `subject`
  - `body_text`
  - `notes_link`
  - `http_link`
  - `owner_department`
  - `category_1`
  - `category_2`
  - `category_3`
  - `category_4`
  - `category_5`
  - `file_path`
  - `source_file_modified_at`
- 補足:
  - `source_doc_id` は `doc_id` 生成の元になる
  - `body_text` は chunk 分割対象の本文
  - カテゴリは `category_1` から `category_5` に展開して持たせる
  - `file_path` と `source_file_modified_at` は各 row 共通値でも、正規化後の列として持たせる

### 1.2 サンプル JSON からの対応表

- `doc_id` -> `source_doc_id` の元データとして使う
- `metadata.subject` -> `subject`
- `body_text` -> `body_text`
- `metadata.notes_link` -> `notes_link`
- `metadata.http_link` -> `http_link`
- `metadata.owner_department` -> `owner_department`
- `metadata.categories[0]` -> `category_1`
- `metadata.categories[1]` -> `category_2`
- `metadata.categories[2]` -> `category_3`
- `metadata.categories[3]` -> `category_4`
- `metadata.categories[4]` -> `category_5`
- ファイルシステムから取得 -> `file_path`, `source_file_modified_at`

### 1.3 chunk 対象テキストの初期方針

- 初期方針は `body_text` のみを chunk 対象にする。
- `subject` は検索精度のために本文前置したくなる可能性があるが、まずは metadata として保持するだけに留める。
- 必要なら後続タスクとして「`subject + body_text` を chunk 元テキストにするか」を再検討する。

### 2. XML由来JSON 用の内部データモデルを決める

- JSON 1 record を Python 上で扱うための中間表現を決める。
- 最低限保持する項目を決める。
- 候補:
  - record 固有 ID
  - subject
  - body_text
  - http_link
  - notes_link
  - owner_department
  - categories
  - 元ファイルパス
  - 元 XML ファイル更新日時

### 3. metadata 設計を整理する

- CSV と XML の両方で使う metadata の共通項目を洗い出す。
- CSV 専用 metadata と XML 専用 metadata を分離する方針を決める。
- `BaseDocument.metadata` に何を保存するか定義する。
- 宿題:
  - CSV の `file_path` / 更新日時 / 行 ID と、XML由来JSON の `source_doc_id` / 元 JSON ファイル更新日時 / `notes_link` をどう統一的に表現するか設計する。
  - `doc_id` の命名規則を CSV と XML由来JSON でどう揃えるか決める。

### 4. XML由来JSON レコードを取り込み可能にする

- 方針A: `process_file()` 系に JSON/JSONL Reader を追加する。
- 方針B: JSON/JSONL を DataFrame に正規化した上で、`process_dataframe()` 系を「行内 chunk 分割対応」に拡張して使う。
- 現状コードとの整合では、まず方針Bの方が差分が小さい。
- 具体タスク:
  - JSON/JSONL のレコード列挙処理を実装する。
  - 1 record -> 1 row の正規化処理を実装する。
  - `body_text` を chunk 元カラムとして渡せるようにする。
  - `source_doc_id` を `id_col_name` に使えるようにする。

### 5. DataFrame 取り込みで chunk 分割できるようにする

- `process_dataframe()` は現状 1 行 = 1 chunk なので、今回の要件には不足している。
- 1 record を起点にしつつ、`body_text` を `chunk_processor` で複数 chunk に分割できるようにする。
- 既存の `process_dataframe()` の意味は変えず、chunk 分割対応は別メソッドで追加する。
- API の第一候補:
  - `process_dataframe_with_chunking(df, chunk_col_name, id_col_name, source_file_path=None, source_file_modified_at=None)`
- 具体タスク:
  - `DocumentProcessingService` に `process_dataframe_with_chunking()` のような別メソッドを追加する。
  - `DataFrameDocumentProcessingPipeline` を拡張するか、chunk 分割専用の新しい pipeline を追加する。
  - 各 row について `body_text` を chunk 分割し、1 row から複数 document を生成できるようにする。
  - 各 chunk に対して、元 row 共通の metadata を引き継ぐようにする。
  - chunk ごとに既存の `serial_number` を連番として使う。
  - `doc_id` を `source_doc_id + chunk番号` のように一意化する。

### 5.1 別メソッドの引数案を固める

- 最低限必要な引数:
  - `df`
  - `chunk_col_name`
  - `id_col_name`
- 任意引数候補:
  - `record_type_col_name`
  - `title_col_name`
- 方針:
  - record 本文の chunk 分割そのものは service が持つ `self._chunk_processor` を使う。
  - 呼び出し側に chunk processor を毎回渡させない。
  - 元 JSON ファイル由来の共通情報も、DataFrame 列として渡す。

### 5.3 `process_dataframe_with_chunking()` の使い方イメージ

- 呼び出し例のイメージ:

```python
doc_service.process_dataframe_with_chunking(
    df=json_df,
    chunk_col_name="body_text",
    id_col_name="source_doc_id",
)
```

- もし `source_file_path` や `source_file_modified_at` を DataFrame 列に入れないなら、引数でこう渡す案もある。
- `source_file_path` と `source_file_modified_at` は DataFrame 列に含める前提なので、追加引数は不要とする。

### 5.2 `doc_id` 生成ルールを固める

- 第一候補:
  - `{source_doc_id}_No.{serial_number}`
- 例:
  - `4925766C003A0765:F3A1B2C4D5E6F7890123456789ABCDEF_No.0`
  - `4925766C003A0765:F3A1B2C4D5E6F7890123456789ABCDEF_No.1`
- 決めるべきこと:
  - record_id に `:` を含んでも問題ないか。
  - UI 表示上長すぎる場合に別の表示名が必要か。
  - 既存 `process_file()` の `{filename}_No.{serial_number}` と完全に揃えるか、source ごとに許容するか。

### 6. 行内 chunk 分割時の metadata 仕様を決める

- `process_file()` は `serial_number`、`page_number`、`chunk_index_in_page` を持っている。
- XML由来JSON では `page_number` の概念がないため、record 基準の項目に置き換える必要がある。
- 具体タスク:
  - `page_number` を使わない場合の代替 metadata 項目を決める。
  - 候補:
    - `source_doc_id`
    - `record_index`
    - `source_type`
    - `source_file_path`
    - `source_file_modified_at`
  - `process_file()` と `process_dataframe()` 系で共通利用できる metadata 名と、片方専用の metadata 名を分ける。

### 6.1 metadata の初期案

- 共通 metadata:
  - `source_type`
  - `source_file_path`
  - `source_file_modified_at`
  - `text`
- XML由来JSON 専用 metadata:
  - `source_doc_id`
  - `record_index`
  - `subject`
  - `http_link`
  - `notes_link`
  - `owner_department`
  - `category_1`
  - `category_2`
  - `category_3`
  - `category_4`
  - `category_5`
- 取り込み時に `images` と `attachments` は metadata に入れない。
- カテゴリは将来的に Streamlit 側のフィルター条件として使えるよう、`category_1` から `category_5` に展開して保持する。

### 6.2 空を許容する metadata 方針

- metadata は「全ソースで同じ key を持てるなら持つ、存在しない値は空でもよい」という方針でそろえる。
- これにより、既存の `process_file()` 由来データと XML由来JSON を同じ collection に入れても扱いやすくする。
- 想定:
  - 既存ファイル取り込みでは `notes_link` などは空
  - XML由来JSON では `page_number=1`、`chunk_index_in_page=serial_number`
  - CSV 相当データでは `source_doc_id` や `owner_department` は空

### 6.3 共通 metadata と拡張 metadata の考え方

- 既存の `doc_service3` に合わせて、まずは既存 metadata と整合することを優先する。
- その上で、XML由来JSON は持てる metadata を追加で持たせる。
- イメージ:
  - 既存ファイル由来で主に使う項目
    - `file_path`
    - `serial_number`
    - `page_number`
    - `chunk_index_in_page`
  - XML由来JSON で追加する項目
    - `source_doc_id`
    - `notes_link`
    - `http_link`
    - `subject`
    - `owner_department`
    - `category_1` から `category_5`
    - `source_file_modified_at`
- 追加項目は、他のソースで存在しない場合に空でよい。
- XML由来JSON では `file_path` に JSON ファイルパスを入れる。
- XML由来JSON では `page_number=1`、`chunk_index_in_page=serial_number` とする。

### 6.4 `page_number` の扱い

- `page_number` は PDF/Word などページや段落単位で読む既存 `process_file()` には自然だが、XML由来JSON には本質的には存在しない。
- ただし既存実装では `page_number` は `1` 始まりで扱われているため、XML由来JSON でも `page_number = 1` で統一する方針にする。
- `page_number` の意味は実ページではなく、「非ページ系ソースの既定値」として扱う。
- 必要なら `record_index` は別項目として持つ。

### 6.5 `BaseDocument` 項目の現状と今後の整理

| 項目名 | 現在の格納先 | 現在の意味 | 現在の主な利用 | 今後の扱い | XML由来JSONでの値 | 既存ファイル取り込みでの値 |
| --- | --- | --- | --- | --- | --- | --- |
| `doc_id` | `BaseDocument` 直下 | 現状は chunk 単位のID | `process_file`, `process_dataframe` | 実装上は維持。意味としては `chunk_id` 相当と理解する | `{source_doc_id}_No.{chunk_index}` 形式を想定 | 既存通り `{filename}_No.{serial_number}` |
| `text` | `BaseDocument` 直下 | chunk 本文 | `process_file`, `process_dataframe` | 維持 | `body_text` を chunk 分割した本文 | 既存通り本文 chunk |
| `db_name` | `BaseDocument` 直下 | 保存先DB名 | repository内部 | 原則維持 | 現状通り | 現状通り |
| `distance` | `BaseDocument` 直下 | 検索距離 | 検索結果 | 原則維持 | 検索時に設定 | 検索時に設定 |
| `file_path` | `metadata` | 元ファイルのパス | `process_file` | 維持。JSONでも取り込み元ファイルのパスとして使う | JSONファイルのパス | 既存ファイルパス |
| `serial_number` | `metadata` | chunk の通し番号 | `process_file` | 維持。JSONでも各 source_doc 内の chunk 連番として使う | 0,1,2... | 既存通り |
| `page_number` | `metadata` | ページ/段落単位の番号 | `process_file` | 維持。XML由来JSON では既定値 `1` を入れる | 常に `1` | 既存通り |
| `chunk_index_in_page` | `metadata` | ページ内 chunk 番号 | `process_file` | 維持。XML由来JSON では `serial_number` と同じ値を入れる | `serial_number` と同じ値 | 既存通り |
| `source_doc_id` | `metadata` 追加候補 | chunk の親になる元ドキュメントID | 新規 | 追加 | JSON内の元 `doc_id` | ファイル名などファイル由来ID |
| `source_file_modified_at` | `metadata` 追加候補 | 元データファイルの更新日時 | 新規 | 追加 | JSONファイル更新日時 | 必要ならファイル更新日時、未使用なら空 |
| `notes_link` | `metadata` 追加候補 | Notes 参照リンク | 新規 | 追加 | JSONの `metadata.notes_link` | 空 |
| `http_link` | `metadata` 追加候補 | HTTP参照リンク | 新規 | 追加候補 | JSONの `metadata.http_link` | 空 |
| `subject` | `metadata` 追加候補 | 文書タイトル | 新規 | 追加候補 | JSONの `metadata.subject` | 空 |
| `owner_department` | `metadata` 追加候補 | 所管部署 | 新規 | 追加候補 | JSONの `metadata.owner_department` | 空 |
| `category_1` - `category_5` | `metadata` 追加候補 | 分類情報 | 新規 | 追加。将来の Streamlit フィルター候補 | JSONの `metadata.categories` を展開 | 空 |

### 6.6 表の読み方

- 現行互換を優先するため、既存 `BaseDocument` の `doc_id` / `text` / 既存 metadata key は基本的に残す。
- その上で、XML由来JSON のために `source_doc_id` などの拡張 metadata を追加する。
- `doc_id` は名前上は維持するが、意味としては chunk 単位IDであることを明記して扱う。
- `serial_number` は、XML由来JSON でも各 `source_doc_id` ごとに 0 始まりの chunk 連番として使う。
- XML由来JSON では 1 record を 1 page 相当とみなし、`chunk_index_in_page` も `serial_number` と同じ値で扱う。
- ソースごとに存在しない項目は空でよい。

### 6.7 1 record から複数 chunk を作るときの処理イメージ

1. JSON record を 1 row に正規化する
2. `row[chunk_col_name]` を `self._chunk_processor.process()` に渡す
3. 分割された各 chunk について metadata を複製する
4. `serial_number` を付ける
5. `doc_id` を `source_doc_id + chunk番号` で生成する
6. chunks / metadatas をまとめて repository に保存する

### 6.8 具体例: JSON 1 record が 2 chunk になる場合

- 入力 JSON record の例:

```json
{
  "doc_id": "4925766C003A0765:F3A1B2C4D5E6F7890123456789ABCDEF",
  "metadata": {
    "subject": "旅費規程",
    "http_link": "https://example.com/db/4925766C003A0765/0/F3A1B2C4D5E6F7890123456789ABCDEF",
    "notes_link": "notes://server/db/4925766C003A0765/0/F3A1B2C4D5E6F7890123456789ABCDEF",
    "owner_department": "経理部",
    "categories": ["規程", "経理", "旅費"]
  },
  "body_text": "第1章 総則 ... 長文 ..."
}
```

- 正規化後 DataFrame 1 row のイメージ:

```python
{
    "source_doc_id": "4925766C003A0765:F3A1B2C4D5E6F7890123456789ABCDEF",
    "subject": "旅費規程",
    "body_text": "第1章 総則 ... 長文 ...",
    "notes_link": "notes://server/db/4925766C003A0765/0/F3A1B2C4D5E6F7890123456789ABCDEF",
    "http_link": "https://example.com/db/4925766C003A0765/0/F3A1B2C4D5E6F7890123456789ABCDEF",
    "owner_department": "経理部",
    "category_1": "規程",
    "category_2": "経理",
    "category_3": "旅費",
    "category_4": "",
    "category_5": "",
    "file_path": ".../ragpon/examples/xml_example.json",
    "source_file_modified_at": "2026-03-23T10:00:00+09:00",
}
```

- `body_text` が 2 chunk に分かれた場合の保存イメージ:

```python
BaseDocument(
    doc_id="4925766C003A0765:F3A1B2C4D5E6F7890123456789ABCDEF_No.0",
    text="第1章 総則 ... chunk 0 ...",
    metadata={
        "file_path": ".../ragpon/examples/xml_example.json",
        "serial_number": 0,
        "page_number": 1,
        "chunk_index_in_page": 0,
        "source_doc_id": "4925766C003A0765:F3A1B2C4D5E6F7890123456789ABCDEF",
        "source_file_modified_at": "2026-03-23T10:00:00+09:00",
        "notes_link": "notes://server/db/4925766C003A0765/0/F3A1B2C4D5E6F7890123456789ABCDEF",
        "http_link": "https://example.com/db/4925766C003A0765/0/F3A1B2C4D5E6F7890123456789ABCDEF",
        "subject": "旅費規程",
        "owner_department": "経理部",
        "category_1": "規程",
        "category_2": "経理",
        "category_3": "旅費",
        "category_4": "",
        "category_5": "",
    }
)

BaseDocument(
    doc_id="4925766C003A0765:F3A1B2C4D5E6F7890123456789ABCDEF_No.1",
    text="... chunk 1 ...",
    metadata={
        "file_path": ".../ragpon/examples/xml_example.json",
        "serial_number": 1,
        "page_number": 1,
        "chunk_index_in_page": 1,
        "source_doc_id": "4925766C003A0765:F3A1B2C4D5E6F7890123456789ABCDEF",
        "source_file_modified_at": "2026-03-23T10:00:00+09:00",
        "notes_link": "notes://server/db/4925766C003A0765/0/F3A1B2C4D5E6F7890123456789ABCDEF",
        "http_link": "https://example.com/db/4925766C003A0765/0/F3A1B2C4D5E6F7890123456789ABCDEF",
        "subject": "旅費規程",
        "owner_department": "経理部",
        "category_1": "規程",
        "category_2": "経理",
        "category_3": "旅費",
        "category_4": "",
        "category_5": "",
    }
)
```

- ポイント:
  - `source_doc_id` は 2 chunk で共通
  - `doc_id` は chunk ごとに一意
  - `serial_number` と `chunk_index_in_page` は同じ値
  - `page_number` は常に `1`
  - `file_path` は JSON ファイル自体のパス

### 7. ChromaDB 保存内容を確認する

- XML由来JSON metadata が ChromaDB に問題なく保存できるか確認する。
- `notes_link`、`http_link`、更新日時などが検索結果の metadata として復元されることを確認する。
- `category_1` から `category_5` が将来の Streamlit フィルターに使える形で保存されているか確認する。

### 8. FastAPI の検索結果整形を拡張する

- 現状は `doc_id`、`semantic distance`、`Text` しか Streamlit に渡っていない。
- `notes_link` を表示するには、FastAPI 側で system message 用文字列に埋め込むか、別構造で返す必要がある。
- 最小差分なら、context string に `notes_link` 行を追加する。
- 具体タスク:
  - `build_context_string()` に XML metadata の出力を追加する。
  - `parse_context_string_to_structured()` が `notes_link` を読めるようにする。
  - CSV では `notes_link` がないケースを許容する。

### 9. Streamlit の表示を拡張する

- 「回答に使用された情報を見る」に `notes_link` を追加表示する。
- XML由来JSON と CSV で表示項目が崩れないようにする。
- 具体タスク:
  - context row に `notes_link` がある場合のみ表示する。
  - 必要なら `http_link` も今後表示できるよう、表示部を metadata 拡張しやすい形に整理する。

### 10. basic_usage.py に XML由来JSON 取り込み例を追加する

- `doc_service3` 相当の永続化パターンで JSON データ投入例を追加する。
- 例では `process_dataframe_with_chunking()` を使う。
- CSV と XML の両方を同じ collection に入れる例、または collection を分ける例を決める。
- サンプルから検索して metadata が返ることを確認できるコードを用意する。

### 11. テスト観点を追加する

- XML由来JSON レコードの正規化テスト
- 1 row -> 複数 chunk 展開テスト
- chunk ごとの `doc_id` / metadata 一意性テスト
- metadata 生成テスト
- ChromaDB 保存/検索時の metadata 復元テスト
- FastAPI の context string 生成・再パーステスト
- Streamlit 表示ロジックの最低限の回帰確認

## 実装順の推奨

1. 取り込み仕様を固定する
2. metadata 設計を決める
3. JSON record -> DataFrame/内部表現への変換を作る
4. 別メソッドで DataFrame 系 chunk 分割を追加する
5. ChromaDB へ保存できる状態にする
6. FastAPI で `notes_link` を通す
7. Streamlit で表示する
8. `basic_usage.py` とテストを整える

## 設計上の論点

- XML由来JSON 対応を `DocumentReader` 拡張で入れるか、前処理 + `process_dataframe()` 再利用で入れるか。
- chunk 分割対応は `process_dataframe()` を変更せず、別メソッドとして追加する。
- 別メソッドの引数にどこまで source 情報を持たせるか。
- metadata の共通スキーマをどこまで揃えるか。
- `categories` のようなリスト項目を ChromaDB metadata にそのまま載せるか。
- `notes_link` 以外の XML由来 metadata も UI に出すか、今回は最小限に留めるか。

## まず着手すべき最小スコープ

- JSON/JSONL の 1 record を 1 row に変換できるようにする。
- 1 row の `body_text` を複数 chunk に分割して保存できるようにする。
- `body_text` と `notes_link` を metadata 付きで ChromaDB に保存する。
- 検索結果から `notes_link` を Streamlit に表示する。

## 実装チェックリスト

### A. JSON 正規化

- [ ] `xml_example.json` の入力形式を JSON Lines として扱うか最終確定する
- [ ] JSON 1 record を辞書として読み出す処理を作る
- [ ] 必要項目だけを抽出して DataFrame 化する
- [ ] `images` と `attachments` を取り込み対象から除外する
- [ ] `record_id` に使う列を決める
- [ ] `source_doc_id` に使う列を決める
- [ ] `source_file_path` と `source_file_modified_at` の取得方法を決める

### B. Service / Pipeline 拡張

- [ ] `DocumentProcessingService` に `process_dataframe_with_chunking()` を追加する
- [ ] chunk 分割専用 pipeline を新設するか、既存 pipeline を拡張するか確定する
- [ ] 各 row の `body_text` を `self._chunk_processor` で分割する
- [ ] 1 row から複数 chunk と複数 metadata を生成できるようにする
- [ ] `doc_id = {source_doc_id}_No.{serial_number}` で一意化する
- [ ] `serial_number` を各 row 内の chunk 連番として使う
- [ ] row 共通 metadata を各 chunk に引き継ぐ

### C. Metadata 設計

- [ ] 共通 metadata 項目を確定する
- [ ] XML由来JSON 専用 metadata 項目を確定する
- [ ] `categories` をそのまま保存するか文字列化するか決める
- [ ] `source_type` の値を決める
- [ ] `record_index` を持たせるか決める
- [ ] CSV と XML由来JSON で削除・検索時に困らない key 名にそろえる

### D. Repository / 検索確認

- [ ] ChromaDB に metadata が保存できることを確認する
- [ ] 検索結果から `notes_link` とその他 metadata が復元されることを確認する
- [ ] BM25 も使う場合に metadata 互換性が崩れないか確認する

### E. FastAPI

- [ ] `build_context_string()` に `notes_link` を含める
- [ ] 必要なら `http_link` など追加候補の出し方も決める
- [ ] `parse_context_string_to_structured()` が `notes_link` を読めるようにする
- [ ] CSV 由来の結果で `notes_link` が空でも壊れないようにする

### F. Streamlit

- [ ] 「回答に使用された情報を見る」で `notes_link` を表示する
- [ ] `notes_link` がない行では表示をスキップする
- [ ] 将来 metadata 項目が増えても崩れにくい表示にする
- [ ] 将来的な `categories` フィルター追加を見据えて、metadata 参照しやすい構造を維持する

### G. サンプル / 動作確認

- [ ] `basic_usage.py` に JSON 取り込み例を追加する
- [ ] `process_dataframe()` と `process_dataframe_with_chunking()` の使い分けが分かる例にする
- [ ] JSON 取り込み後に検索し、`notes_link` を含む結果を確認する

### H. テスト

- [ ] JSON 正規化テストを追加する
- [ ] 1 row -> 複数 chunk 展開のテストを追加する
- [ ] `doc_id` 生成ルールのテストを追加する
- [ ] metadata 復元テストを追加する
- [ ] FastAPI の context string / parse テストを追加する
- [ ] Streamlit 表示の最低限の回帰確認を行う
