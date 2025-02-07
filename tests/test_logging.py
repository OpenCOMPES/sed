import io
import logging
import os
from datetime import datetime

import pytest

from sed.core.logging import call_logger
from sed.core.logging import set_verbosity
from sed.core.logging import setup_logging


@pytest.fixture
def logger_():
    logger = setup_logging("test_logger")
    log_capture_string = io.StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    yield logger, log_capture_string


def test_debug_logging(logger_):
    logger, log_capture_string = logger_
    logger.debug("This is a debug message")
    assert "This is a debug message" in log_capture_string.getvalue()


def test_info_logging(logger_):
    logger, log_capture_string = logger_
    logger.info("This is an info message")
    assert "This is an info message" in log_capture_string.getvalue()


def test_warning_logging(logger_):
    logger, log_capture_string = logger_
    logger.warning("This is a warning message")
    assert "This is a warning message" in log_capture_string.getvalue()


def test_error_logging(logger_):
    logger, log_capture_string = logger_
    logger.error("This is an error message")
    assert "This is an error message" in log_capture_string.getvalue()


def test_critical_logging(logger_):
    logger, log_capture_string = logger_
    logger.critical("This is a critical message")
    assert "This is a critical message" in log_capture_string.getvalue()


def test_set_verbosity(logger_):
    logger, log_capture_string = logger_
    set_verbosity(logger, verbose=True)
    assert logger.handlers[0].level == logging.INFO
    set_verbosity(logger, verbose=False)
    assert logger.handlers[0].level == logging.WARNING


def test_logger_has_base_logger(logger_):
    logger, log_capture_string = logger_
    assert logger.name == "sed.test_logger"
    assert logger.parent.name == "sed"
    assert logger.parent.parent.name == "root"
    assert logger.parent.level == logging.DEBUG
    assert isinstance(logger.parent.handlers[0], logging.FileHandler)


def test_logger_creates_logfile(tmp_path):
    logger = setup_logging("test_logger", set_base_handler=True, user_log_path=tmp_path)
    log_file = os.path.join(tmp_path, f"sed_{datetime.now().strftime('%Y-%m-%d')}.log")
    assert os.path.exists(log_file)
    with open(log_file) as f:
        assert f.read() == ""
    logger.debug("This is a debug message")
    with open(log_file) as f:
        assert "This is a debug message" in f.read()


def test_readonly_path(tmp_path, caplog):
    os.chmod(tmp_path, 0o444)
    with caplog.at_level(logging.WARNING):
        setup_logging("test_logger", set_base_handler=True, user_log_path=tmp_path)
    assert f"Cannot create logfile in Folder {tmp_path}, disabling logfile." in caplog.messages[0]
    log_file = os.path.join(tmp_path, f"sed_{datetime.now().strftime('%Y-%m-%d')}.log")
    assert not os.path.exists(log_file)
