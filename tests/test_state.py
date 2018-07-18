import unittest

import torchbearer


class TestStateKey(unittest.TestCase):
    def test_key_added(self):
        key = torchbearer.state_key('key')

        self.assertTrue(key in torchbearer.STATE_KEYS)

    def test_duplicate(self):
        key = torchbearer.state_key(torchbearer.MODEL)

        self.assertTrue(torchbearer.MODEL != key)
