from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from app.core.scheduler import start_scheduler, stop_scheduler, get_scheduler


class TestScheduler:

    def setup_method(self):
        stop_scheduler()

    def teardown_method(self):
        stop_scheduler()

    def test_start_and_stop(self):
        scheduler = start_scheduler("0 7 * * *")
        assert scheduler.running
        stop_scheduler()
        assert get_scheduler() is None

    def test_start_idempotent(self):
        s1 = start_scheduler("0 7 * * *")
        s2 = start_scheduler("0 7 * * *")
        assert s1 is s2
        stop_scheduler()

    def test_invalid_cron(self):
        with pytest.raises(ValueError, match="cron"):
            start_scheduler("invalid")

    def test_stop_when_not_running(self):
        stop_scheduler()

    def test_get_scheduler_none_initially(self):
        assert get_scheduler() is None
