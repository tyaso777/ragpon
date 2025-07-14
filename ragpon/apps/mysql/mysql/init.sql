MYSQL_DATABASE=ragpon-mysql;

CREATE TABLE IF NOT EXISTS sessions (
    user_id VARCHAR(255) NOT NULL,
    session_id CHAR(36) NOT NULL,
    app_name TEXT NOT NULL,
    session_name TEXT NOT NULL,
    is_private_session TINYINT(1) DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL,
    updated_at TIMESTAMP NULL DEFAULT NULL,
    updated_by TEXT,
    is_deleted TINYINT(1) DEFAULT 0,
    deleted_by TEXT,
    PRIMARY KEY (user_id, session_id)
) ENCRYPTION='Y';

CREATE TABLE IF NOT EXISTS messages (
    id CHAR(36) PRIMARY KEY,
    round_id INT NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    session_id CHAR(36) NOT NULL,
    app_name TEXT NOT NULL,
    content TEXT NOT NULL,
    message_type VARCHAR(10) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,
    updated_at TIMESTAMP NULL DEFAULT NULL,
    updated_by TEXT,
    is_deleted TINYINT(1) DEFAULT 0,
    deleted_by TEXT,
    llm_model TEXT,
    use_chroma TINYINT(1) DEFAULT 0,
    use_bm25 TINYINT(1) DEFAULT 0,
    use_reranker TINYINT(1) DEFAULT 0,
    rerank_model TEXT,
    file_name TEXT,
    file_type TEXT,
    file_content TEXT,
    feedback TEXT,
    feedback_at TIMESTAMP NULL DEFAULT NULL,
    feedback_reason TEXT,
    rag_mode TEXT,
    CONSTRAINT uq_messages_round_role
      UNIQUE (user_id, session_id, round_id, message_type),
    CHECK (message_type IN ('system', 'user', 'assistant'))
) ENCRYPTION='Y';