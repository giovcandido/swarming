import logging

from sys import stdout as sys_stdout
from time import strftime

class Logger(logging.Logger):

    LOGGING_FORMAT = '[%(asctime)s] %(name)17s - %(levelname)5s - %(message)s'

    LOGGING_FILE = 'task-%s.log' % (strftime('%Y%m%d-%H%M%S'))

    def get_logger(logger_name):
        # Set this class as the logger
        logging.setLoggerClass(Logger)

        # Get the logger instance
        logger = logging.getLogger(logger_name)

        # Set the default logger level
        logger.setLevel(logging.DEBUG)

        # Create a stream handler
        stream_handler = logging.StreamHandler(sys_stdout)
        stream_handler.setFormatter(logging.Formatter(Logger.LOGGING_FORMAT))

        # Create a file handler
        file_handler = logging.FileHandler(Logger.LOGGING_FILE)
        file_handler.setFormatter(logging.Formatter(Logger.LOGGING_FORMAT))

        # Add handlers to logger instance
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

        return logger

    def write(self, message):
        # Deactivate stream handler
        self.handlers[0].setLevel(logging.CRITICAL)

        # Write message to file
        self.info(message)

        # Reactivate stream handler
        self.handlers[0].setLevel(logging.INFO)
