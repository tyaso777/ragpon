# %%
import importlib.resources as ir
import logging

import pandas as pd

from ragpon import (
    Config,
    DocumentProcessingService,
    JAGinzaChunkProcessor,
    RuriLargeEmbedder,
    RuriRerankerLargeEvaluator,
)

# この例では "ragpon.examples"フォルダに格納されているサンプルファイルを利用する。
with ir.as_file(ir.files("ragpon.examples")) as examples_dir:
    pdf_path = examples_dir / "投資信託とは.pdf"
    word_path = examples_dir / "ragponの使い方.docx"
    pdf_config_path = examples_dir / "sample_config_for_pdf.yml"
    df_config_path = examples_dir / "sample_config_for_dataframe.yml"
    saving_db_config_path = examples_dir / "sample_config_for_saving_db.yml"

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
config = Config(pdf_config_path)
print(config.config)

doc_service = DocumentProcessingService(config_or_config_path=config)

# %%time
# pdfとwordファイルのDBへの格納
doc_service.process_file(str(pdf_path))
doc_service.process_file(str(word_path))

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
config2 = Config(df_config_path)
config2.config

# %%
doc_service_2 = DocumentProcessingService(config_or_config_path=config2)

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
config3 = Config(saving_db_config_path)
config3.config


# %%
chunk_processor = JAGinzaChunkProcessor(chunk_size=300)

# %%
doc_service3 = DocumentProcessingService(
    config_or_config_path=config3,
    embedder=RuriLargeEmbedder(config=config),
    chunk_processor=chunk_processor,
    relevance_evaluator=RuriRerankerLargeEvaluator(config=config),
)

# %%
# データの格納（in-memoryではなくファイルとして永続化する）
# doc_service3.process_file(str(pdf_path))
# doc_service3.process_file(str(word_path))

# ここで使うデータベースには既にこのセルの処理をしている。
# ここでは、再度のデータ追加せずにデータ取得ができるかを確認する。

# %%
search_results = doc_service3.search("投資信託のリスク")
enhanced_results = doc_service3.enhance_search_results(search_results)
reranked_results = doc_service3.rerank_results(
    query="投資信託のリスク", search_results=enhanced_results
)
reranked_results

# %%
# ところで、CPUだとrerankの処理に17秒かかっており、ちょっと遅すぎな感じがある。
# やはりGPUであるが、CPUの高速化もあるようだ。バージョンが合わなかったが調整の余地はありそう。https://pypi.org/project/ctranslate2/

# %%
