# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import unittest

from monai.apps.auto3dseg.auto_runner import logger


class TestLogger(unittest.TestCase):
    def test_vista3d_logger(self):
        from scripts.train import CONFIG

        logging.config.dictConfig(CONFIG)
        logger.warning("check train logging format")

    def test_vista3d_logger_infer(self):
        from scripts.infer import CONFIG

        logging.config.dictConfig(CONFIG)
        logger.warning("check infer logging format")


if __name__ == "__main__":
    unittest.main()
