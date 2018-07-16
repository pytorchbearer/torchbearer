import unittest

import sconce


class TestStateKey(unittest.TestCase):
    def test_key_added(self):
        key = sconce.state_key('key')

        self.assertTrue(key in sconce.STATE_KEYS)

    def test_duplicate(self):
        key = sconce.state_key(sconce.MODEL)

        self.assertTrue(sconce.MODEL != key)
