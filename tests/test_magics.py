import unittest
from mock import Mock, patch


class TestMagics(unittest.TestCase):
    @patch('IPython.core.magic.register_line_magic')
    def test_magic_normal(self, mock_magic):
        stored_fn = lambda x: None

        def my_func(fn):
            nonlocal stored_fn
            stored_fn = fn

        mock_magic.side_effect = my_func
        import magics
        self.assertTrue(magics.is_notebook())
        stored_fn('normal')
        self.assertFalse(magics.is_notebook())
        stored_fn('notebook')
        self.assertTrue(magics.is_notebook())
        del magics

        mock_magic.side_effect = NameError
        import magics
