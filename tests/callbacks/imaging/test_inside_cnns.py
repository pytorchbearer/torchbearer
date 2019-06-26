from unittest import TestCase

from mock import MagicMock, patch, ANY

import torch

import torchbearer
import torchbearer.callbacks.imaging as imaging


class TestClassAppearanceModel(TestCase):
    def test_targets_hot(self):
        callback = imaging.ClassAppearanceModel(nclasses=10, input_size=(1, 1, 1), target=5)
        targets_hot = callback._targets_hot({torchbearer.DEVICE: 'cpu'})
        self.assertTrue(targets_hot.sum() == 1)
        self.assertTrue(targets_hot[0][5] == 1)

    def test_targets_to_key(self):
        callback = imaging.ClassAppearanceModel(nclasses=10, input_size=(1, 1, 1), target=5)
        callback = callback.target_to_key('test')
        state = {torchbearer.DEVICE: 'cpu'}
        targets_hot = callback._targets_hot(state)
        self.assertTrue(targets_hot.sum() == 1)
        self.assertTrue(targets_hot[0][5] == 1)
        self.assertTrue(state['test'] == 5)

    @patch('torchbearer.callbacks.imaging.inside_cnns.torch.randint')
    def test_random_targets(self, randint):
        randint.return_value = torch.ones(1, 1) * 6
        callback = imaging.ClassAppearanceModel(nclasses=10, input_size=(1, 1, 1))
        state = {torchbearer.DEVICE: 'cpu'}
        targets_hot = callback._targets_hot(state)
        self.assertTrue(targets_hot.sum() == 1)
        self.assertTrue(targets_hot[0][6] == 1)

    def test_cam_loss(self):
        key = 'my_key'
        targets_hot = torch.FloatTensor([[0, 0, 0, 1, 0]]).ge(0.5)
        decay = 0.5
        loss = imaging.inside_cnns._cam_loss(key, targets_hot, decay)

        model = MagicMock
        model.input_image = torch.ones(5, 5) * 2

        state = {'my_key': torch.FloatTensor([[0, 0, 0, 100, 0]]), torchbearer.MODEL: model}
        res = loss(state)
        self.assertTrue(res.item() == -50)

    def test_cam_wrapper(self):
        model = MagicMock()
        wrapper = imaging.inside_cnns._CAMWrapper((3, 10, 10), model)

        self.assertTrue(wrapper.input_image.requires_grad)
        self.assertTrue(wrapper.input_image.shape == torch.Size([3, 10, 10]))
        wrapper('', 'state')
        self.assertTrue(model.call_count == 1)
        self.assertTrue(model.call_args[0][0].shape == torch.Size([1, 3, 10, 10]))
        self.assertTrue(model.call_args[0][1] == 'state')
        model.reset_mock()

        model.side_effect = lambda x: x
        wrapper('', 'state')
        self.assertTrue(model.call_count == 2)
        self.assertTrue(model.call)
        self.assertTrue(model.call_args[0][0].shape == torch.Size([1, 3, 10, 10]))

    @patch('torchbearer.callbacks.imaging.inside_cnns._cam_loss')
    @patch('torchbearer.callbacks.imaging.inside_cnns._CAMWrapper')
    @patch('torchbearer.callbacks.imaging.inside_cnns.torchbearer.Trial')
    def test_on_batch(self, _, wrapper, loss):
        wrapper().input_image = torch.nn.Parameter(torch.zeros(10))

        factory = MagicMock()
        callback = imaging.ClassAppearanceModel(nclasses=10, input_size=(1, 1, 1), optimizer_factory=factory)

        model = MagicMock()
        callback.on_batch({torchbearer.EPOCH: 0, torchbearer.MODEL: model, torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.float32})

        loss.assert_called_once_with(torchbearer.PREDICTION, ANY, 0.01)

        self.assertTrue(model.eval.call_count == 1)
        self.assertTrue(model.train.call_count == 1)
        self.assertTrue(factory.call_count == 1)
        self.assertTrue(next(iter(factory.call_args[0][0])) is wrapper().input_image)

    def test_end_to_end(self):
        with torchbearer.no_grad():
            model = torch.nn.Linear(10, 5)
            callback = imaging.ClassAppearanceModel(nclasses=5, input_size=(10), steps=1)
            state = {torchbearer.EPOCH: 0, torchbearer.MODEL: model, torchbearer.DEVICE: 'cpu',
                     torchbearer.DATA_TYPE: torch.float32}
            callback.on_batch(state)
