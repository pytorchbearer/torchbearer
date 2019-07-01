import unittest
import warnings

import torchbearer
from torchbearer.state import State


class TestStateKey(unittest.TestCase):
    def test_key_metric(self):
        key = torchbearer.state_key('test')
        state = {key: 4}

        self.assertDictEqual(key.process(state), {str(key): 4})
        self.assertDictEqual(key.process_final(state), {str(key): 4})

    def test_key_call(self):
        key = torchbearer.state_key('call_test')
        state = {key: 'test'}

        self.assertEqual(key(state), 'test')

    def test_key_repr(self):
        key = torchbearer.state_key('repr_test')
        self.assertEqual(str(key), 'repr_test')
        self.assertEqual(repr(key), 'repr_test')

    def test_key_added(self):
        key = torchbearer.state_key('key')
        self.assertTrue('key' in torchbearer.state.__keys__)

    def test_collision(self):
        _ = torchbearer.state_key('test')
        key_1 = torchbearer.state_key('test')
        key_2 = torchbearer.state_key('test')

        self.assertTrue('test' != str(key_1))
        self.assertTrue('test' != str(key_2))

    def test_duplicate_string(self):
        _ = torchbearer.state_key('test_dup')
        key_1 = torchbearer.state_key('test_dup')
        key_2 = torchbearer.state_key('test_dup')

        self.assertTrue('test_dup_1' == str(key_1))
        self.assertTrue('test_dup_2' == str(key_2))

    def test_compare_to_statekey(self):
        key_1 = torchbearer.state_key('test_compare_sk')
        key_2 = torchbearer.state_key('test_compare_sk_2')
        # Simulates same key in different sessions where the object hash is changed
        key_2.key = 'test_compare_sk'
        self.assertEqual(key_1, key_2)

    def test_compare_to_string(self):
        key_1 = torchbearer.state_key('test_compare')
        self.assertEqual(key_1, 'test_compare')


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

    def test_warn(self):
        s = State()

        key1 = torchbearer.state_key('test_a')
        key2 = torchbearer.state_key('test_b')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            s[key1] = 'key_1'
            s[key2] = 'key_2'
            s['bad_key'] = 'bad_key'
            self.assertTrue(len(w) == 1)
            self.assertTrue('State was accessed with a string' in str(w[-1].message))
