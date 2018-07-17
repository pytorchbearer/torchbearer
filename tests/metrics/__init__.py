import sys
from unittest.mock import MagicMock


class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


sys.modules.update([('torch.cuda.current_device', Mock())])
