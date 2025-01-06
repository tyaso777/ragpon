import logging


def get_library_logger(name: str) -> logging.Logger:
    """
    Retrieve a logger with the specified name and ensure it has a NullHandler by default.

    Args:
        name (str): The name of the logger (usually `__name__`).

    Returns:
        logging.Logger: A configured logger instance with a NullHandler by default.
    """
    logger = logging.getLogger(name)
    # Add NullHandler only if it doesn't exist
    if not any(isinstance(handler, logging.NullHandler) for handler in logger.handlers):
        logger.addHandler(logging.NullHandler())
    return logger
