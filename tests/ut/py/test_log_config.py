# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for simpler_setup.log_config — off must mute everything."""

import logging
import os

from simpler_setup.log_config import (
    DEFAULT_LOG_LEVEL,
    LOG_LEVEL_CHOICES,
    configure_logging,
)


def test_off_mutes_all_records(caplog):
    configure_logging("off")
    logger = logging.getLogger("simpler.test.log_config")
    with caplog.at_level(logging.NOTSET, logger=logger.name):
        logger.error("err-should-be-muted")
        logger.critical("crit-should-be-muted")
    assert caplog.records == []
    assert os.environ["PTO_LOG_LEVEL"] == "off"


def test_error_level_still_emits_error(caplog):
    configure_logging("error")
    logger = logging.getLogger("simpler.test.log_config")
    with caplog.at_level(logging.ERROR, logger=logger.name):
        logger.error("err-must-show")
    assert any("err-must-show" in r.message for r in caplog.records)


def test_choices_contains_off_only():
    assert "off" in LOG_LEVEL_CHOICES
    assert "silent" not in LOG_LEVEL_CHOICES
    assert DEFAULT_LOG_LEVEL == "info"
