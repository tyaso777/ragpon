# %%
def _get_math_instruction() -> str:
    return (
        "When presenting mathematical expressions, follow this rule:\n"
        "- If the user explicitly requests plaintext math notation (e.g., in Excel-style or code-style), preserve that format.\n"
        "- Otherwise, always write math expressions using LaTeX syntax.\n"
        "- Use `$...$` for inline math, and `$$...$$` for block math.\n"
        "- Use LaTeX commands such as `\\frac`, `\\cdot`, `\\sum`, `\\mathbf` appropriately.\n"
        "\n"
    )


def _get_response_style_instruction() -> str:
    return "Be concise, accurate, and consistent with formatting."


def _get_rag_citation_instruction() -> str:
    return (
        "If your answer is based on relevant documents, you MUST always cite the reference materials that support your statements using their RAG Rank.\n"
        "Use the format: [[RAG_RANK=number]].\n"
        "This format is REQUIRED so the system can later extract references.\n"
        "If multiple sources are used, include all relevant RAG Rank values like [[RAG_RANK=1]], [[RAG_RANK=2]].\n"
        "Do NOT include doc_id or semantic distance.\n"
        "Every factual statement based on retrieved content MUST include the RAG Rank.\n"
        "Example:\n"
        "1. 強化されたテキストは、検索文脈を明確にするのに役立ちます [[RAG_RANK=5]]。\n"
        "2. 様々なテキストを扱えることで柔軟なインプットを可能にしています [[RAG_RANK=8]]。\n"
        "\n"
    )

def _get_vector_search_guidelines() -> str:
    return (
        "Your task is to reconstruct the user's request from the conversation above "
        "so it can effectively retrieve the most relevant documents using vector search in **Japanese**. \n"
        "Currently, only vector search is used (not BM25+), so it is essential that your reformulated queries maximize semantic relevance.\n"
        "\n"
        "Please focus on generating queries that will help answer the **final user question** in the conversation. "
        "You may refer to the context of the entire conversation to understand the user's intent, "
        "but your output should support answering the final question directly and concretely.\n"
        "\n"
        "If needed, split the request into **1 to 3 queries**.\n"
    )


def _get_strict_json_format_instruction() -> str:
    return (
        "You must respond **only** with strictly valid JSON, containing an array of objects, "
        "each with a single 'query' key. No extra text or explanation is allowed.\n"
        "For example:\n"
        "[\n"
        "    {\"query\": \"Example query 1\"},\n"
        "    {\"query\": \"Example query 2\"}\n"
        "]\n"
        "If only one query is sufficient, you may include just one object.\n"
        "Ensure that each query captures the user's intent clearly and naturally, using language that enhances the effectiveness of vector-based retrieval."
    )




def get_system_prompt_no_rag() -> str:
    return (
        "You are a helpful assistant. Please answer in Japanese.\n"
        f"{_get_math_instruction()}"
        f"{_get_response_style_instruction()}"
    )


def get_system_prompt_with_context() -> str:
    return (
        "You are a helpful assistant. Please answer in Japanese.\n"
        f"{_get_rag_citation_instruction()}"
        f"{_get_math_instruction()}"
        f"{_get_response_style_instruction()}"
    )


def get_optimized_query_instruction() -> str:
    return (
        f"{_get_vector_search_guidelines()}"
        f"{_get_strict_json_format_instruction()}"
    )



SYSTEM_PROMPT_NO_RAG = get_system_prompt_no_rag() 

SYSTEM_PROMPT_WITH_CONTEXT = get_system_prompt_with_context()

OPTIMIZED_QUERY_INSTRUCTION = """
Your task is to reconstruct the user's request from the conversation above so it can effectively retrieve the most relevant documents using vector search in **Japanese**. 
Currently, only vector search is used (not BM25+), so it is essential that your reformulated queries maximize semantic relevance.

Please focus on generating queries that will help answer the **final user question** in the conversation. You may refer to the context of the entire conversation to understand the user's intent, but your output should support answering the final question directly and concretely.

If needed, split the request into **1 to 3 queries**. You must respond **only** with strictly valid JSON, containing an array of objects, each with a single 'query' key. No extra text or explanation is allowed. For example:
[
    {"query": "Example query 1"},
    {"query": "Example query 2"}
]
If only one query is sufficient, you may include just one object. Ensure that each query captures the user's intent clearly and naturally, using language that enhances the effectiveness of vector-based retrieval.
"""

# %%
