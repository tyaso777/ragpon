CREATE TABLE IF NOT EXISTS sessions (
    user_id TEXT NOT NULL,
    session_id UUID NOT NULL,
    app_name TEXT NOT NULL,
    session_name TEXT NOT NULL,
    is_private_session BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL,
    updated_at TIMESTAMP,
    updated_by TEXT,
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_by TEXT,
    PRIMARY KEY (user_id, session_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY,
    round_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    session_id UUID NOT NULL,
    app_name TEXT NOT NULL,
    content TEXT NOT NULL,
    message_type VARCHAR(10) NOT NULL CHECK (message_type IN ('system', 'user', 'assistant')),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,
    updated_at TIMESTAMP,
    updated_by TEXT,
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_by TEXT,
    llm_model TEXT,
    use_chroma BOOLEAN DEFAULT FALSE,
    use_bm25 BOOLEAN DEFAULT FALSE,
    use_reranker BOOLEAN DEFAULT FALSE,
    rerank_model TEXT,
    file_name TEXT,
    file_type TEXT,
    file_content TEXT,
    feedback TEXT,
    feedback_at TIMESTAMP,
    feedback_reason TEXT,
    rag_mode TEXT,
    CONSTRAINT uq_messages_round_role
      UNIQUE (user_id, session_id, round_id, message_type)
);
