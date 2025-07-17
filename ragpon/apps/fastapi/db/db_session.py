from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import mysql.connector
import psycopg2.extensions
from mysql.connector import errors as mysql_errors
from mysql.connector.pooling import MySQLConnectionPool
from psycopg2 import Error as Psycopg2Error
from psycopg2.errors import UniqueViolation
from psycopg2.pool import SimpleConnectionPool

from ragpon._utils.logging_helper import get_library_logger
from ragpon.apps.fastapi.db.db_errors import (
    DatabaseConflictError,
    DatabaseQueryError,
    DatabaseUnavailableError,
)

# Initialize logger
logger = get_library_logger(__name__)


class DatabaseSession(ABC):
    """Abstract base class for managing database sessions.

    This interface defines a context manager for executing SQL queries
    and fetching results in a backend-agnostic way.

    Methods:
        __enter__: Enters the context manager and returns the session object.
        __exit__: Exits the context manager, handling commit/rollback.
        execute: Executes an SQL query with optional parameters.
        fetchall: Retrieves all rows from the last executed query.
        fetchone: Retrieves a single row from the last executed query.

    Properties:
        rowcount: The number of rows affected by the last executed query.

    """

    @abstractmethod
    def __enter__(self) -> "DatabaseSession":
        """Enter the context manager and return the database session.

        Returns:
            DatabaseSession: The session instance.
        """
        pass

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the context manager, committing or rolling back as needed.

        Args:
            exc_type: The type of exception raised (if any).
            exc_val: The exception instance (if any).
            exc_tb: The traceback object (if any).
        """
        pass

    @abstractmethod
    def execute(self, query: str, params: tuple = ()) -> None:
        """Executes a SQL query.

        Args:
            query: SQL query string.
            params: Optional query parameters.
        """
        pass

    @abstractmethod
    def fetchall(self) -> list[tuple]:
        """Fetches all rows from the last query.

        Returns:
            list[tuple]: A list of rows as tuples.
        """
        pass

    @abstractmethod
    def fetchone(self) -> tuple | None:
        """Fetches a single row from the last query.

        Returns:
            tuple | None: A single row or None if no results.
        """
        pass

    @property
    @abstractmethod
    def rowcount(self) -> int:
        """
        Returns the number of rows affected by the last execute call.

        Returns:
            int: Number of rows affected.
        """
        pass


class PostgreDBSession(DatabaseSession):
    """PostgreSQL-specific implementation of DatabaseSession using psycopg2.

    This class wraps a psycopg2 connection and cursor inside a context manager
    and automatically handles commit or rollback on exit.

    Attributes:
        pool (SimpleConnectionPool): The psycopg2 connection pool.
    """

    def __init__(self, pool: SimpleConnectionPool):
        """Initializes the PostgreDBSession.

        Args:
            pool: psycopg2 SimpleConnectionPool instance.
        """
        self.pool = pool
        self.conn: psycopg2.extensions.connection | None = None
        self.cursor: psycopg2.extensions.cursor | None = None

    def __enter__(self) -> "PostgreDBSession":
        """Acquires a connection and cursor from the pool.

        Returns:
            PostgreDBSession: The active database session.

        Raises:
            DatabaseUnavailableError: If the database connection fails.
        """
        try:
            self.conn = self.pool.getconn()
            self.cursor = self.conn.cursor()
            return self
        except Psycopg2Error as e:
            logger.exception("[PostgreDBSession] Failed to obtain connection:")
            raise DatabaseUnavailableError("PostgreSQL connection failed") from e

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Commits or rolls back the transaction and returns the connection.

        Args:
            exc_type: The type of exception raised (if any).
            exc_val: The exception instance (if any).
            exc_tb: The traceback object (if any).
        """
        if self.cursor:
            self.cursor.close()
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
                logger.exception(
                    "[PostgreDBSession] Transaction rolled back due to exception."
                )
            self.pool.putconn(self.conn)

    def execute(self, query: str, params: tuple = ()) -> None:
        """Execute a SQL query with optional parameters.

        Args:
            query: SQL query string.
            params: Optional tuple of parameters.

        Raises:
            RuntimeError: If the cursor is not initialized.
        """
        if self.cursor is None:
            logger.error(
                "[PostgreDBSession] execute() called but cursor is not initialized"
            )
            raise RuntimeError(
                "[PostgreDBSession] Cursor not initialized. Use within a 'with' context."
            )
        try:
            self.cursor.execute(query, params)
        except UniqueViolation as e:
            logger.error(
                "[PostgreDBSession] Unique constraint violation", exc_info=True
            )
            raise DatabaseConflictError("Unique constraint violation") from e
        except Psycopg2Error as e:
            logger.error("[PostgreDBSession] Query execution failed", exc_info=True)
            raise DatabaseQueryError("PostgreSQL query failed") from e

    def fetchall(self) -> list[tuple]:
        """Fetch all rows from the last query result.

        Returns:
            list[tuple]: A list of tuples representing rows.

        Raises:
            RuntimeError: If the cursor is not initialized.
        """
        if self.cursor is None:
            logger.error(
                "[PostgreDBSession] fetchall() called but cursor is not initialized"
            )
            raise RuntimeError(
                "[PostgreDBSession] Cursor not initialized. Use within a 'with' context."
            )
        return self.cursor.fetchall()

    def fetchone(self) -> tuple | None:
        """Fetch one row from the last query result.

        Returns:
            tuple | None: A single result row, or None if no data.

        Raises:
            RuntimeError: If the cursor is not initialized.
        """
        if self.cursor is None:
            logger.error(
                "[PostgreDBSession] fetchone() called but cursor is not initialized"
            )
            raise RuntimeError(
                "[PostgreDBSession] Cursor not initialized. Use within a 'with' context."
            )
        return self.cursor.fetchone()

    @property
    def rowcount(self) -> int:
        if self.cursor is None:
            logger.error(
                "[PostgreDBSession] fetchone() called but cursor is not initialized"
            )
            raise RuntimeError(
                "[PostgreDBSession] Cursor not initialized. Use within a 'with' context."
            )
        return self.cursor.rowcount


class MySQLDBSession(DatabaseSession):
    """MySQL implementation using mysql-connector-python.

    This class wraps a MySQL connection and cursor inside a context manager
    and automatically handles commit or rollback on exit.

    Attributes:
        pool (MySQLConnectionPool): The mysql-connector connection pool.
    """

    def __init__(self, pool: MySQLConnectionPool) -> None:
        """Initializes the MySQLDBSession.

        Args:
            pool: mysql.connector.pooling.MySQLConnectionPool instance.
        """
        self.pool = pool
        self.conn: mysql.connector.MySQLConnection | None = None
        self.cursor: mysql.connector.cursor.MySQLCursor | None = None

    def __enter__(self) -> MySQLDBSession:
        """Acquires a connection and cursor from the pool.

        Returns:
            MySQLDBSession: The active database session.

        Raises:
            DatabaseUnavailableError: If the database connection fails.
        """
        try:
            self.conn = self.pool.get_connection()
            self.cursor = self.conn.cursor(buffered=True)
            return self
        except mysql_errors.Error as e:
            logger.exception("[MySQLDBSession] Failed to obtain connection:")
            raise DatabaseUnavailableError("MySQL connection failed") from e

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Commits or rolls back the transaction and releases the connection.

        Args:
            exc_type: The type of exception raised (if any).
            exc_val: The exception instance (if any).
            exc_tb: The traceback object (if any).
        """
        if self.cursor:
            self.cursor.close()

        if self.conn is None:
            return

        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
                logger.exception(
                    "[MySQLDBSession] Transaction rolled back due to exception"
                )
        finally:
            self.conn.close()

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        """Execute a SQL query with optional parameters.

        Args:
            query: SQL query string.
            params: Optional tuple of parameters.

        Raises:
            RuntimeError: If the cursor is not initialized.
            mysql.connector.Error: If the query execution fails.
        """
        if self.cursor is None:
            logger.error(
                "[MySQLDBSession] execute() called but cursor is not initialized"
            )
            raise RuntimeError(
                "[MySQLDBSession] Cursor not initialized. Use within a 'with' context."
            )

        try:
            self.cursor.execute(query, params)
        except mysql_errors.IntegrityError as e:
            logger.error(
                "[MySQLDBSession] Integrity constraint violation", exc_info=True
            )
            raise DatabaseConflictError("Integrity constraint violation") from e
        except mysql_errors.Error as e:
            logger.error("[MySQLDBSession] Query execution failed", exc_info=True)
            raise DatabaseQueryError("MySQL query failed") from e

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Fetch all rows from the last query result.

        Returns:
            list[tuple[Any, ...]]: A list of tuples representing rows.

        Raises:
            RuntimeError: If the cursor is not initialized.
        """
        if self.cursor is None:
            logger.error(
                "[MySQLDBSession] fetchall() called but cursor is not initialized"
            )
            raise RuntimeError(
                "[MySQLDBSession] Cursor not initialized. Use within a 'with' context."
            )
        return self.cursor.fetchall()

    def fetchone(self) -> tuple[Any, ...] | None:
        """Fetch one row from the last query result.

        Returns:
            tuple[Any, ...] | None: A single result row, or None if no data.

        Raises:
            RuntimeError: If the cursor is not initialized.
        """
        if self.cursor is None:
            logger.error(
                "[MySQLDBSession] fetchone() called but cursor is not initialized"
            )
            raise RuntimeError(
                "[MySQLDBSession] Cursor not initialized. Use within a 'with' context."
            )
        return self.cursor.fetchone()

    @property
    def rowcount(self) -> int:
        """Get number of rows affected by the last query.

        Returns:
            int: The number of rows affected.

        Raises:
            RuntimeError: If the cursor is not initialized.
        """
        if self.cursor is None:
            logger.error(
                "[MySQLDBSession] rowcount accessed but cursor is not initialized"
            )
            raise RuntimeError(
                "[MySQLDBSession] Cursor not initialized. Use within a 'with' context."
            )
        return self.cursor.rowcount


def get_database_client(db_type: str, pool: Any) -> DatabaseSession:
    """
    Returns an appropriate DatabaseSession based on db_type.

    Args:
        db_type: Type of database backend ("postgres", "mysql", etc.)
        pool: Connection pool object specific to the backend.

    Returns:
        DatabaseSession: A context manager for database operations.
    """
    if db_type == "postgres":
        return PostgreDBSession(pool)
    elif db_type == "mysql":
        return MySQLDBSession(pool)
    else:
        raise ValueError(f"[get_database_client] Unsupported database type: {db_type}")
