class DatabaseError(Exception):
    """Base class for all database-related errors."""


class DatabaseConflictError(DatabaseError):
    """Raised when a conflict occurs, e.g. duplicate primary key or unique constraint."""


class DatabaseUnavailableError(DatabaseError):
    """Raised when the database is unreachable or the connection pool is exhausted."""


class DatabaseQueryError(DatabaseError):
    """Raised for general SQL execution failures not covered by other exceptions."""
