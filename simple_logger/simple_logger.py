import logging
import os


class __SimpleLogger(logging.Logger):
    argument_level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO,
                          "WARNING": logging.WARN,
                          "ERROR": logging.ERROR, "FATAL": logging.FATAL}

    def __init__(self, logger_name: str = ""):
        super(self.__class__, self).__init__(name=logger_name)

        self.setLevel(logging.DEBUG)

        self.__file_logging_level = logging.INFO
        self.__console_logging_level = logging.INFO

        self.__log_file_name = None
        self.__initialized = False

    def init(self):
        if self.__initialized:
            raise RuntimeError("Logger can only be initialized once")
        self.__initialized = True
        if self.__log_file_name is not None:
            self.__create_file_handler()
        self.__create_console_handler()

    def set_file_logging_level(self, logging_level: str):

        log_level_num = -1
        try:
            log_level_num = self.argument_level_map[logging_level.upper()]
        except KeyError:
            print("Unrecognized log level <{}>.".format(logging_level))
            print("Available log levels: {}".format(
                sorted([l for l in self.argument_level_map.keys()],
                       key=lambda l: self.argument_level_map[l])))
            exit(1)
        self.__file_logging_level = log_level_num

    def set_log_file(self, log_file_name: str):
        self.__log_file_name = log_file_name

    def set_console_logging_level(self, logging_level: str):
        log_level_num = -1
        try:
            log_level_num = self.argument_level_map[logging_level.upper()]
        except KeyError:
            print("Unrecognized console log level <{}>.".format(logging_level))
            print("Available log levels: {}".format(
                sorted([l for l in self.argument_level_map.keys()],
                       key=lambda l: self.argument_level_map[l])))
            exit(1)
        self.__console_logging_level = log_level_num

    def __create_console_handler(self):
        ch = logging.StreamHandler()
        ch.setLevel(self.__console_logging_level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S')
        ch.setFormatter(formatter)

        self.handlers = [h for h in self.handlers if type(h) is not logging.StreamHandler]
        self.addHandler(ch)

    def __create_file_handler(self):

        fh = logging.FileHandler(filename=self.__log_file_name)
        fh.setLevel(self.__file_logging_level)

        level_format = "[{levelname:8}]"
        filename_format = "{filename:.15}"
        func = "{funcName:.15}()"
        line = "{lineno:<3}"
        formatter = logging.Formatter(
            level_format + "[" + filename_format + " - " + func + " - " + line + ": " + "{message}",
            style="{")
        fh.setFormatter(formatter)

        self.handlers = [h for h in self.handlers if type(h) is not logging.FileHandler]
        self.addHandler(fh)

        # Create a symlink to faster access.
        log_dir = os.path.dirname(self.__log_file_name)
        last_log_file = os.path.join(log_dir, "__last_log.log")
        if os.path.exists(last_log_file):
            os.remove(last_log_file)
        try:
            os.symlink(self.__log_file_name, last_log_file)
        except OSError:
            pass


SimpleLogger = __SimpleLogger(logger_name="SimpleLogger")
