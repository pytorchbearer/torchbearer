import unittest


class TestMagics(unittest.TestCase):
    def test_torchbearer_fn(self):
        from torchbearer.magics import torchbearer as tb
        from torchbearer.magics import is_notebook

        self.assertFalse(is_notebook())
        tb('notebook')
        self.assertTrue(is_notebook())
        tb('normal')
        self.assertFalse(is_notebook())

