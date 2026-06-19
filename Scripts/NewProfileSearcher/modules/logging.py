from enum import Enum


class LoggingLevel(Enum):
    OFF = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4


# to be fair, I'd hook up KTT's Logger somehow instead
# but that would make the logger public API?
class Logger:
    def __init__(self, level: LoggingLevel, prefix: str) -> None:
        self.level = level
        self.prefix = prefix

    def _Log(self, level: LoggingLevel, message: str):
        if self.level.value < level.value:
            return

        print(_getLoggingLevelString(level), self.prefix, message)

    def Error(self, message: str):
        self._Log(LoggingLevel.ERROR, message)

    def Warning(self, message: str):
        self._Log(LoggingLevel.WARNING, message)

    def Info(self, message: str):
        self._Log(LoggingLevel.INFO, message)

    def Debug(self, message: str):
        self._Log(LoggingLevel.DEBUG, message)


def _getLoggingLevelString(level: LoggingLevel):
    match level:
        case LoggingLevel.OFF:
            return '[Off]'
        case LoggingLevel.ERROR:
            return '[Error]'
        case LoggingLevel.WARNING:
            return '[Warning]'
        case LoggingLevel.INFO:
            return '[Info]'
        case LoggingLevel.DEBUG:
            return '[Debug]'

        case _:
            raise NameError('Unhandled logging level value')
            return ''
