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
