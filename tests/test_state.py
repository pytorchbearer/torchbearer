import unittest

import bink


class TestStateKey(unittest.TestCase):
    def test_key_added(self):
        key = bink.state_key('key')

        self.assertTrue(key in bink.STATE_KEYS)

    def test_duplicate(self):
        key = bink.state_key(bink.MODEL)

        self.assertTrue(bink.MODEL != key)
