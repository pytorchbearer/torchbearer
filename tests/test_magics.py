import unittest
from mock import Mock, patch


class TestMagics(unittest.TestCase):
    @patch('IPython.core.magic.register_line_magic')
    def test_magic_normal(self, mock_magic):
        stored_fn = {'fn': lambda x: None}

        def my_func(fn):
            stored_fn['fn'] = fn

        mock_magic.side_effect = my_func
        import torchbearer.magics as magics
        self.assertTrue(magics.is_notebook())
        stored_fn['fn']('normal')
        self.assertFalse(magics.is_notebook())
        stored_fn['fn']('notebook')
        self.assertTrue(magics.is_notebook())
        del magics

        mock_magic.side_effect = NameError
        import torchbearer.magics as magics
