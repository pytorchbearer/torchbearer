import unittest

import torchbearer


class TestStateKey(unittest.TestCase):
    def test_key_added(self):
        key = torchbearer.state_key('key')

        self.assertTrue('key' in torchbearer.state.__keys__)

    def test_duplicate(self):
        _ = torchbearer.state_key('test')
        key_1 = torchbearer.state_key('test')
        key_2 = torchbearer.state_key('test')

        self.assertTrue('test' != key_1)
        self.assertTrue('test' != key_2)
