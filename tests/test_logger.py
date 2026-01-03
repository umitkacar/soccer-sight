"""
Tests for logging module.
"""

import pytest
import logging


class TestLoggerSetup:
    """Tests for logger setup."""

    def test_setup_logger_returns_logger(self):
        """setup_logger should return a Logger instance."""
        from logger import setup_logger

        logger = setup_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_handlers(self):
        """Logger should have at least one handler."""
        from logger import setup_logger

        logger = setup_logger("test_handlers")
        assert len(logger.handlers) >= 1

    def test_get_logger_creates_module_logger(self):
        """get_logger should create module-specific logger."""
        from logger import get_logger

        logger = get_logger("mymodule")
        assert "futbl.mymodule" in logger.name

    def test_convenience_functions_exist(self):
        """Convenience logging functions should be importable."""
        from logger import debug, info, warning, error, critical

        # Should not raise ImportError
        assert callable(debug)
        assert callable(info)
        assert callable(warning)
        assert callable(error)
        assert callable(critical)


class TestLoggerOutput:
    """Tests for logger output."""

    def test_logger_can_log_info(self, caplog):
        """Logger should be able to log info messages."""
        from logger import get_logger

        logger = get_logger("test_info")
        with caplog.at_level(logging.INFO):
            logger.info("Test info message")

        assert "Test info message" in caplog.text

    def test_logger_can_log_error(self, caplog):
        """Logger should be able to log error messages."""
        from logger import get_logger

        logger = get_logger("test_error")
        with caplog.at_level(logging.ERROR):
            logger.error("Test error message")

        assert "Test error message" in caplog.text
