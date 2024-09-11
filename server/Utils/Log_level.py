import logging
from enum import Enum

class LogLevel(str, Enum):
    """
    Enum class for logging levels.
    """
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'


def set_log_level(level: str, logger):
    """
    Set the logging level for the logger.
    """
    logger.setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(level)
    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger("uvicorn").setLevel(level)
    logger.info(f"Logging level set to {level}")
