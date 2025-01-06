# %%
import logging

from ragpon._utils.logging_helper import get_library_logger


# %%
def test_get_library_logger_creates_logger():
    """
    Test that `get_library_logger` creates a logger with the specified name.
    """
    logger_name = "test_logger"
    logger = get_library_logger(logger_name)

    assert isinstance(logger, logging.Logger)
    assert logger.name == logger_name


# %%
def test_get_library_logger_adds_null_handler():
    """
    Test that `get_library_logger` adds a NullHandler if no handlers are present.
    """
    logger_name = "test_logger_with_null_handler"
    logger = get_library_logger(logger_name)

    # Check if a NullHandler is added
    assert any(isinstance(handler, logging.NullHandler) for handler in logger.handlers)


# %%
def test_get_library_logger_does_not_duplicate_handlers():
    """
    Test that `get_library_logger` does not add duplicate handlers.
    """
    logger_name = "test_logger_no_duplicate_handlers"
    # First call to get_library_logger
    logger = get_library_logger(logger_name)

    # Call get_library_logger again
    logger = get_library_logger(logger_name)

    # Ensure only one NullHandler is present
    null_handler_count = sum(
        isinstance(h, logging.NullHandler) for h in logger.handlers
    )
    assert (
        null_handler_count == 1
    ), f"Duplicate NullHandlers detected: {null_handler_count}"
