import unittest

import torchbearer
from torchbearer.state import State

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

class TestState(unittest.TestCase):
    def test_contains(self):
        s = State()

        key1 = torchbearer.state_key('test_a')
        key2 = torchbearer.state_key('test_b')

        s[key1] = 1
        s[key2] = 2

        self.assertTrue(s.__contains__(key1))

    def test_delete(self):
        s = State()

        key1 = torchbearer.state_key('test_a')
        key2 = torchbearer.state_key('test_b')

        s[key1] = 1
        s[key2] = 2

        self.assertTrue(s.__contains__(key1))
        s.__delitem__(key1)
        self.assertFalse(s.__contains__(key1))

    def test_update(self):
        s = State()

        key1 = torchbearer.state_key('test_a')
        key2 = torchbearer.state_key('test_b')

        new_s = {key1: 1, key2: 2}
        s.update(new_s)

        self.assertTrue(s.__contains__(key1))
        self.assertTrue(s[key1] == 1)

    def test_update_state(self):
        s = State()
        new_s = State()

        key1 = torchbearer.state_key('test_a')
        key2 = torchbearer.state_key('test_b')

        new_s_dict = {key1: 1, key2: 2}
        new_s.update(new_s_dict)

        s.update(new_s)
        self.assertTrue(s.__contains__(key1))
        self.assertTrue(s[key1] == 1)