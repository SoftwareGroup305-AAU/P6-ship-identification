import atexit
import colorlog
import collections.abc
import copy
import functools
import inspect
import logging
import logging.config
import queue
import re
import sys
import threading


def get_logger(name):
    """
    Add caller module name as prefix to "name" parameter and
    return default logging.getLogger
    """
    module_name = inspect.stack()[1].frame.f_globals["__name__"]
    return logging.getLogger(f"{module_name}.{name}")


class LogProducer:
    @property
    def logger(self):
        """
        Retrieve the caller class name
        """
        spec = sys.modules[self.__class__.__module__].__spec__
        name = self.__class__.__qualname__
        if spec is not None:
            name = f"{spec.name}.{name}"

        logger = logging.getLogger(name)
        if isinstance(logger, Logger):
            logger.setProducer(self)
        return logger

    def process_log_record(self, log_record):
        """
        LogProducer sub-classes may override this method to transform and/or add
        log record attributes.

        Note: logness.setup() MUST have been called prior to any logger object instanciation.
        """
        log_record.producer = self
        return log_record


class Logger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level=level)
        self._producer = None

    def setProducer(self, producer):
        self._producer = producer

    @functools.wraps(logging.Logger.makeRecord)
    def makeRecord(self, *args, **kwds):
        producer = self._get_producer()
        record = super().makeRecord(*args, **kwds)
        if producer is not None:
            record = producer.process_log_record(record)
        return record

    def _get_producer(self):
        c = self
        rv = None
        while c:
            rv = getattr(c, "_producer", None)
            if rv is not None:
                break
            if not c.propagate:
                break
            else:
                c = c.parent
        return rv


class _DefaultFieldFormatterMixin:

    _arg_pattern = re.compile(r'%\((\w+)\)')

    def format(self, record: logging.LogRecord) -> str:
        arg_names = (x.group(1) for x in self._arg_pattern.finditer(self._fmt))
        for field in arg_names:
            if field not in record.__dict__:
                record.__dict__[field] = ""
        return super().format(record)


class _DefaultFormatter(_DefaultFieldFormatterMixin, logging.Formatter):
    pass


class _ColoredFormatter(_DefaultFieldFormatterMixin, colorlog.ColoredFormatter):
    pass


class DispatchingFormatter:
    def __init__(self, formatter):
        self._formatter = formatter

    def format(self, record):
        if hasattr(record, "_format_"):
            return record._format_(self._formatter, record)
        else:
            return self._formatter.format(record)


def DefaultFormatter(*args, **kwds):
    return DispatchingFormatter(_DefaultFormatter(*args, **kwds))


def ColoredFormatter(*args, **kwds):
    return DispatchingFormatter(_ColoredFormatter(*args, **kwds))


class BackgroundLogHandlerMixin:

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self._background_log_record_queue = queue.Queue()
        self._background_log_thread = threading.Thread(target=self._background_log_handle)
        atexit.register(self._background_log_stop)
        self._background_log_thread.daemon = True
        self._background_log_running = True
        self._background_log_thread.start()

    def emit(self, record):
        self._background_log_record_queue.put_nowait(record)

    def _background_log_handle(self):
        while self._background_log_running:
            try:
                record = self._background_log_record_queue.get(timeout=0.001)
                super().emit(record)
            except queue.Empty:
                pass
        while not self._background_log_record_queue.empty():
            record = self._background_log_record_queue.get_nowait()
            super().emit(record)

    def close(self):
        self._background_log_stop()
        return super().close()

    def _background_log_stop(self):
        self._background_log_running = False
        if self._background_log_thread.is_alive():
            self._background_log_thread.join()


class StreamHandler(BackgroundLogHandlerMixin, colorlog.StreamHandler):
    pass


class FileHandler(BackgroundLogHandlerMixin, logging.FileHandler):
    pass


def setup():
    """
    Setup the `logness.Logger` class as the logging `Logger` class.
    """
    logging.setLoggerClass(Logger)


default_format = "%(asctime)s [%(levelname)s] %(name)s - %(funcName)s - %(message)s"

color_format = (
    "%(asctime)s %(log_color)s[%(levelname)s] "
    "%(reset)s\t%(name)s - %(funcName)s - %(message)s"
)


_config = {
    "version": 1,
    "formatters": {
        "color_formatter": {
            "()": "logness.ColoredFormatter",
            "format": color_format,
        },
        "default_formatter": {
            "()": "logness.DefaultFormatter",
            "format": default_format
        },
    },
    "handlers": {
        "console": {"class": "logness.StreamHandler", "formatter": "color_formatter"}
    },
}

_on_update_cb = set()


def get_config():
    """
    Returns the current logging configuration dictionary as previously set or
    updated by :py:func:`~olympe.log.update_config`.

    See: `Logging config dictionary schema <https://docs.python.org/3/library/logging.config.html#logging-config-dictschema>`_
    """  # noqa
    global _config
    return _config


def _update_dict_recursive(res, update):
    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping):
            res[k] = _update_dict_recursive(res.get(k, {}), v)
        else:
            res[k] = v
    return res


def update_config(update, on_update=None):
    """
    Update (recursively) the current logging configuration dictionary.

    See: `Logging config dictionary schema <https://docs.python.org/3/library/logging.config.html#logging-config-dictschema>`_

    """  # noqa
    global _config
    global _on_update_cb
    new_config = copy.deepcopy(_config)
    _update_dict_recursive(new_config, update)
    new_config["disable_existing_loggers"] = False
    logging.config.dictConfig(new_config)
    if on_update is not None:
        _on_update_cb.add(on_update)

    for cb in _on_update_cb:
        cb()
    _config = new_config
