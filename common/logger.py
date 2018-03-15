# -*- coding: utf-8 -*-

import logging
import sys

logger = logging.getLogger("AppName")
formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)s %(levelname)-8s: %(message)s')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter

logger.addHandler(console_handler)
logger.setLevel(logging.INFO)