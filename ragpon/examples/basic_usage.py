# %%
import importlib.resources as ir
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from ragpon import (
    Config,
    DocumentProcessingService,
    JAGinzaChunkProcessor,
    RuriLargeEmbedder,
    RuriLargeEmbedderCTranslate2,
    RuriRerankerLargeEvaluator,
)

# この例では "ragpon.examples"フォルダに格納されているサンプルファイルを利用する。
with ir.as_file(ir.files("ragpon.examples")) as examples_dir:
    pdf_path = examples_dir / "投資信託とは.pdf"
    word_path = examples_dir / "ragponの使い方.docx"
    xml_json_path = examples_dir / "xml_example.json"
    config_path = examples_dir / "sample_config.yml"


def load_xml_json_as_dataframe(json_path: str | Path) -> pd.DataFrame:
    """Normalize XML-derived JSONL into a DataFrame for chunked ingestion."""
    path = Path(json_path)
    modified_at = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()
    rows: list[dict[str, object]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record.get("metadata", {})
            categories = list(metadata.get("categories", []))
            category_columns = {
                f"category_{i + 1}": (categories[i] if i < len(categories) else "")
                for i in range(5)
            }
            rows.append(
                {
                    "source_doc_id": record["doc_id"],
                    "subject": metadata.get("subject", ""),
                    "body_text": record.get("body_text", ""),
                    "notes_link": metadata.get("notes_link", ""),
                    "http_link": metadata.get("http_link", ""),
                    "owner_department": metadata.get("owner_department", ""),
                    "file_path": str(path),
                    "source_file_modified_at": modified_at,
                    "page_number": 1,
                    **category_columns,
                }
            )

    return pd.DataFrame(rows)

# %%
# 任意でロギングを設定する
logging.basicConfig(
    level=logging.INFO,  # INFOの代わりに.. DEBUG: より詳細なログ, WARNING: 重要ログのみ
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# %%
# 例1:
# - BM25, CHROMADBともにPATH指定なくin-memoryでのDB接続
config = Config(config_path)
config.config

# %%
doc_service = DocumentProcessingService(
    config_or_config_path=config,
    chunk_processor=JAGinzaChunkProcessor(chunk_size=300),
)

# %%time
# pdfとwordファイルのDBへの格納
doc_service.process_file(str(pdf_path))
doc_service.process_file(str(word_path))

# %%
# 検索する
# doc_service.search(query="ragponとは？")

doc_service.enhance_search_results(doc_service.search(query="ragponとは？"), num_before=1, num_after=1)

# %%
# 検索する
doc_service.search(query="投資信託のリスク")

# %%
# 検索する
doc_service.search("検索機能の利用")

# %%
# idによりDBからデータを削除する
doc_service.delete_by_ids(ids=["投資信託とは_No.7", "投資信託とは_No.0"])
doc_service.search("投資信託のリスク")  # 結果の表示
# => No.7, No.0が検索結果からも消えている。

# %%
# metadataによりDBからデータを削除する
doc_service.delete_by_metadata(metadata={"serial_number": 8})
doc_service.search("投資信託のリスク")  # 結果の表示
# => No.8が検索結果からも消えている。

# %%
# 削除したデータを含めてデータを再格納する
doc_service.process_file(str(pdf_path))

# %%
# 検索結果を変数に格納する
search_results = doc_service.search(
    query="検索機能の利用", top_k=20
)  # top_kの設定により上位20件ずつを表示。ただしデータが少ない場合は20件に満たない場合もある。
search_results  # 結果の表示

# 検索結果のtextに前2文、後ろ3文の文章を結合する
enhanced_results = doc_service.enhance_search_results(
    search_results=search_results, num_before=2, num_after=3
)
enhanced_results  # 結果の表示
# Document - enhanced_text項目に追記される

# %%
# 検索結果をrerank modelでrerankする。rerankerは
reranked_results = doc_service.rerank_results(
    query="検索機能の利用",
    search_results=enhanced_results,
    search_result_text_key="enhanced_text",
)
reranked_results  # 結果の表示
# Document - rerank項目にスコアが追記される。このスコアは高いほど良い。
# reranked_resultsはスコア順にソートされている。

# %%
# sample DataFrame
data = {
    "index": [i for i in range(9)],
    "日付": [f"2024-01-0{i}" for i in range(1, 10)],
    "担当者名": [
        "山田太郎",
        "佐藤花子",
        "鈴木一郎",
        "田中美咲",
        "山田太郎",
        "佐藤花子",
        "鈴木一郎",
        "田中美咲",
        "山田太郎",
    ],
    "訪問記録": [
        "新規顧客A社を訪問。商品説明を実施し、次回提案の予定を調整。",
        "既存顧客B社を訪問。要望ヒアリングを行い、追加見積もりを提出予定。",
        "新規顧客C社を訪問。初回提案書を提示し、前向きなフィードバックを得た。",
        "既存顧客D社を訪問。定期フォローアップを実施。契約更新の話題あり。",
        "新規顧客E社を訪問。競合他社の比較資料を要求された。",
        "既存顧客F社を訪問。トラブル対応のため追加サポートを約束。",
        "新規顧客G社を訪問。興味を示すが、予算の問題で検討中とのこと。",
        "既存顧客H社を訪問。製品アップデートの説明会を実施。",
        "新規顧客I社を訪問。要望に合わせたカスタマイズ提案を準備する予定。",
    ],
}

df = pd.DataFrame(data)
df

# %%
config = Config(config_path)
config.set("DATABASES.CHROMADB_COLLECTION_NAME", "dataframe_collection")
config.config

# %%
doc_service_2 = DocumentProcessingService(config_or_config_path=config)

# %%
doc_service_2.process_dataframe(df=df, chunk_col_name="訪問記録", id_col_name="index")

# %%
search_results2 = doc_service_2.search(query="商品の要望")
search_results2

# %%
# dataframeの場合は文章をchunkで分割しないためenhanced_resultsは存在しない
# enhanced_results = doc_service_2.enhance_search_results(search_results=search_results)
# enhanced_results

# %%
reranked_results2 = doc_service_2.rerank_results(
    query="商品の要望", search_results=search_results2, search_result_text_key="text"
)
reranked_results2


# %%
# パターン3:
# - フォルダにデータを格納：config設定
# - 文章の区切りサイズを長文に：chunk_processorの設定
# - embedding modelの変更: RuriLargeEmbedderの利用

# %%
config = Config(config_path)
config.set(
    "DATABASES.BM25_PATH",
    "D:\\Users\\AtsushiSuzuki\\OneDrive\\デスクトップ\\test\\ragpon\\ragpon\\examples\\db\\bm25",
)
config.set(
    "DATABASES.CHROMADB_FOLDER_PATH",
    "D:\\Users\\AtsushiSuzuki\\OneDrive\\デスクトップ\\test\\ragpon\\ragpon\\examples\\db",
)
config.config

# %%
chunk_processor = JAGinzaChunkProcessor(chunk_size=300)

# %%
doc_service3 = DocumentProcessingService(
    config_or_config_path=config,
    embedder=RuriLargeEmbedder(config=config),
    # embedder=RuriLargeEmbedderCTranslate2(config=config),
    chunk_processor=chunk_processor,
    relevance_evaluator=RuriRerankerLargeEvaluator(config=config),
)

# %%
# データの格納（in-memoryではなくファイルとして永続化する）
doc_service3.process_file(str(pdf_path))
doc_service3.process_file(str(word_path))

# XML由来JSONの各recordを1 source_docとしてchunk分割しながら格納する
xml_df = load_xml_json_as_dataframe(xml_json_path)
xml_df[
    [
        "source_doc_id",
        "subject",
        "owner_department",
        "category_1",
        "category_2",
        "category_3",
    ]
]
doc_service3.process_dataframe_with_chunking(
    df=xml_df,
    chunk_col_name="body_text",
    id_col_name="source_doc_id",
)

# ここで使うデータベースには既にこのセルの処理をしている。
# ここでは、再度のデータ追加せずにデータ取得ができるかを確認する。

# %%
search_results = doc_service3.search("投資信託のリスク")
enhanced_results = doc_service3.enhance_search_results(search_results)
# %%
reranked_results = doc_service3.rerank_results(
    query="投資信託のリスク", search_results=enhanced_results
)
reranked_results

# %%
# XML由来JSONから入れたデータの検索例
xml_search_results = doc_service3.search("在宅勤務の申請手続")
xml_enhanced_results = doc_service3.enhance_search_results(xml_search_results)
xml_reranked_results = doc_service3.rerank_results(
    query="在宅勤務の申請手続", search_results=xml_enhanced_results
)
xml_reranked_results

# %%
# notes_link や categories を含む metadata を確認する
xml_reranked_results["chroma"][0].base_document.metadata

# %%
# ところで、CPUだとrerankの処理に17秒かかっており、ちょっと遅すぎな感じがある。
# やはりGPUであるが、CPUの高速化もあるようだ。バージョンが合わなかったが調整の余地はありそう。https://pypi.org/project/ctranslate2/

# %%
