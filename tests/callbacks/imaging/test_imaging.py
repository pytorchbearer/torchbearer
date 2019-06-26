from unittest import TestCase

from mock import MagicMock, patch, ANY

import torch

import torchbearer
import torchbearer.callbacks.imaging as imaging

import matplotlib.pyplot as plt  # Import so that it can be mocked
plt.ioff()


class TestHandlers(TestCase):
    @patch('PIL.Image')
    def test_to_file(self, pil):
        handler = imaging.imaging._to_file('test')
        mock = MagicMock()
        handler(mock, 0, '')
        mock.mul.assert_called_once_with(255)
        mock.mul().clamp.assert_called_once_with(0, 255)
        self.assertTrue(mock.mul().clamp().byte.call_count == 1)
        mock.mul().clamp().byte().permute.assert_called_once_with(1, 2, 0)
        self.assertTrue(mock.mul().clamp().byte().permute().cpu.call_count == 1)
        self.assertTrue(mock.mul().clamp().byte().permute().cpu().numpy.call_count == 1)

        pil.fromarray.assert_called_once_with(mock.mul().clamp().byte().permute().cpu().numpy())
        pil.fromarray().save.assert_called_once_with('test')

    @patch('matplotlib.pyplot')
    def test_to_pyplot(self, plt):
        handler = imaging.imaging._to_pyplot(title='test')
        mock = MagicMock()
        handler(mock, 0, '')
        mock.mul.assert_called_once_with(255)
        mock.mul().clamp.assert_called_once_with(0, 255)
        self.assertTrue(mock.mul().clamp().byte.call_count == 1)
        mock.mul().clamp().byte().permute.assert_called_once_with(1, 2, 0)
        self.assertTrue(mock.mul().clamp().byte().permute().cpu.call_count == 1)
        self.assertTrue(mock.mul().clamp().byte().permute().cpu().numpy.call_count == 1)

        plt.title.assert_called_once_with('test')

        plt.imshow.assert_called_once_with(mock.mul().clamp().byte().permute().cpu().numpy())
        self.assertTrue(plt.show.call_count == 1)

    @patch('torchbearer.callbacks.tensor_board')
    def test_to_tensorboard(self, tboard):
        handler = imaging.imaging._to_tensorboard('test', log_dir='./logs', comment='comment')
        image = MagicMock()
        handler(image, 0, {torchbearer.EPOCH: 1})
        image.clamp.assert_called_once_with(0, 1)
        tboard.get_writer.assert_called_once_with('./logs/comment', imaging.imaging._to_tensorboard)
        tboard.get_writer().add_image.assert_called_once_with('test', image.clamp(), 1)
        tboard.close_writer.assert_called_once_with('./logs/comment', imaging.imaging._to_tensorboard)

    @patch('torchbearer.callbacks.tensor_board')
    def test_to_visdom(self, tboard):
        handler = imaging.imaging._to_visdom('test', log_dir='./logs', comment='comment', visdom_params='test_params')
        image = MagicMock()
        handler(image, 0, {torchbearer.EPOCH: 1})
        image.clamp.assert_called_once_with(0, 1)
        tboard.get_writer.assert_called_once_with('./logs/comment', imaging.imaging._to_visdom, visdom=True, visdom_params='test_params')
        tboard.get_writer().add_image.assert_called_once_with('test_1', image.clamp(), 1)
        tboard.close_writer.assert_called_once_with('./logs/comment', imaging.imaging._to_visdom)


class TestImagingCallback(TestCase):
    def test_transform(self):
        callback = imaging.ImagingCallback()
        self.assertTrue(callback.transform('test') is 'test')

        callback = imaging.ImagingCallback(transform=lambda _: 'test')
        self.assertTrue(callback.transform('something else') is 'test')

    @patch('torchbearer.callbacks.imaging.imaging._cache_images')
    def test_cache(self, mock_cache_images):
        callback = imaging.ImagingCallback()
        callback.cache(10)
        mock_cache_images.assert_called_once_with(10)

    @patch('torchvision.utils.make_grid')
    def test_make_grid(self, mock_grid):
        callback = imaging.ImagingCallback()
        callback.on_batch = lambda _: 10
        callback.make_grid(1, 2, True, 4, True, 6)
        callback.on_batch({})
        mock_grid.assert_called_once_with(10, nrow=1, padding=2, normalize=True, range=4, scale_each=True, pad_value=6)

    def test_process(self):
        callback = imaging.ImagingCallback()
        callback.on_batch = MagicMock()
        handler = MagicMock()
        callback = callback.with_handler(handler)
        callback.transform = MagicMock()
        state = 'state'
        callback.process(state)
        self.assertTrue(callback.transform.call_count == 1)
        handler.assert_called_once_with(callback.transform()[None], 0, 'state')
        self.assertTrue(callback.transform().dim.call_count == 1)

    @patch('torchbearer.callbacks.imaging.imaging._to_visdom')
    @patch('torchbearer.callbacks.imaging.imaging._to_tensorboard')
    @patch('torchbearer.callbacks.imaging.imaging._to_pyplot')
    @patch('torchbearer.callbacks.imaging.imaging._to_file')
    def test_simple_methods(self, mock_to_file, mock_to_pyplot, mock_to_tensorboard, mock_to_visdom):
        callback = imaging.ImagingCallback()
        self.assertRaises(NotImplementedError, lambda: callback.on_batch('test'))

        callback = imaging.ImagingCallback().to_file('test')
        self.assertTrue(mock_to_file.call_count == 1)

        callback = imaging.ImagingCallback().to_pyplot()
        self.assertTrue(mock_to_pyplot.call_count == 1)

        callback = imaging.ImagingCallback().to_state('test')
        state = {}
        callback._handlers[0][0]('image', 0, state)
        self.assertTrue('test' in state)
        self.assertTrue(state['test'] is 'image')

        callback = imaging.ImagingCallback().to_state(0)
        state = {}
        callback._handlers[0][0]('image', 0, state)
        self.assertTrue(0 in state)
        self.assertTrue(state[0] is 'image')

        callback = imaging.ImagingCallback().to_tensorboard()
        self.assertTrue(mock_to_tensorboard.call_count == 1)

        callback = imaging.ImagingCallback().to_visdom()
        self.assertTrue(mock_to_visdom.call_count == 1)

    @patch('torchbearer.callbacks.imaging.imaging._to_file')
    def test_index(self, mock_to_file):
        callback = imaging.ImagingCallback().to_file('test', index=10)
        image = torch.zeros(11, 1, 1, 1)
        callback.on_batch = lambda _: image
        callback.process('state')
        mock_to_file().assert_called_once_with(ANY, 10, 'state')
        self.assertTrue((mock_to_file().call_args_list[0][0][0] == image[10]).all())
        mock_to_file().reset_mock()

        callback = imaging.ImagingCallback().to_file('test', index=[2, 5, 10])
        image = torch.rand(11, 1, 1, 1)
        callback.on_batch = lambda _: image
        callback.process('state')
        self.assertTrue(mock_to_file().call_count == 3)
        self.assertTrue((mock_to_file().call_args_list[0][0][0] == image[2]).all())
        self.assertTrue((mock_to_file().call_args_list[1][0][0] == image[5]).all())
        self.assertTrue((mock_to_file().call_args_list[2][0][0] == image[10]).all())

    @patch('torchbearer.callbacks.imaging.imaging._to_file')
    def test_dims(self, mock_to_file):
        callback = imaging.ImagingCallback().to_file('test')
        image = torch.rand(1, 1, 1)
        callback.on_batch = lambda _: image
        callback.process('state')
        mock_to_file().assert_called_once_with(ANY, 0, 'state')
        self.assertTrue((mock_to_file().call_args_list[0][0][0] == image).all())
        self.assertTrue(mock_to_file().call_args_list[0][0][0].dim() == 3)

    def test_on_train(self):
        callback = imaging.ImagingCallback()
        mock = MagicMock()
        callback.on_step_training = mock
        callback.process = MagicMock()
        callback = callback.on_train()
        callback.on_step_training('state')
        mock.assert_called_once_with('state')
        callback.process.assert_called_once_with('state')

    def test_on_val(self):
        callback = imaging.ImagingCallback()
        mock = MagicMock()
        callback.on_step_validation = mock
        callback.process = MagicMock()
        callback = callback.on_val()
        state = {torchbearer.DATA: torchbearer.TEST_DATA}
        callback.on_step_validation(state)
        mock.assert_called_once_with(state)
        self.assertTrue(callback.process.call_count == 0)

        mock.reset_mock()
        callback.process.reset_mock()

        state = {torchbearer.DATA: torchbearer.VALIDATION_DATA}
        callback.on_step_validation(state)
        mock.assert_called_once_with(state)
        callback.process.assert_called_once_with(state)

    def test_on_test(self):
        callback = imaging.ImagingCallback()
        mock = MagicMock()
        callback.on_step_validation = mock
        callback.process = MagicMock()
        callback = callback.on_test()
        state = {torchbearer.DATA: torchbearer.VALIDATION_DATA}
        callback.on_step_validation(state)
        mock.assert_called_once_with(state)
        self.assertTrue(callback.process.call_count == 0)

        mock.reset_mock()
        callback.process.reset_mock()

        state = {torchbearer.DATA: torchbearer.TEST_DATA}
        callback.on_step_validation(state)
        mock.assert_called_once_with(state)
        callback.process.assert_called_once_with(state)


class TestCachingImagingCallback(TestCase):
    def test_main(self):
        callback = imaging.CachingImagingCallback(key='my_key', num_images=5)
        self.assertRaises(NotImplementedError, lambda: callback.on_cache('', {}))

        callback.on_cache = MagicMock()
        callback.on_cache.return_value = 'image'

        state = {'my_key': torch.ones(10, 3, 2, 2), torchbearer.BATCH: 0}
        callback.on_batch(state)
        self.assertTrue(callback.on_cache.call_args[0][0].sum() == 60)
        self.assertTrue(callback.on_cache.call_args[0][1] == state)

    def test_multi_batch(self):
        callback = imaging.CachingImagingCallback(key='my_key', num_images=25)
        self.assertRaises(NotImplementedError, lambda: callback.on_cache('', {}))

        callback.on_cache = MagicMock()
        callback.on_cache.return_value = 'image'

        state = {'my_key': torch.ones(10, 3, 2, 2), torchbearer.BATCH: 0}
        callback.on_batch(state)
        state[torchbearer.BATCH] = 1
        callback.on_batch(state)
        state[torchbearer.BATCH] = 2
        callback.on_batch(state)
        self.assertTrue(callback.on_cache.call_args[0][0].sum() == 300)
        self.assertTrue(callback.on_cache.call_args[0][1] == state)

    def test_multi_epoch(self):
        callback = imaging.CachingImagingCallback(key='my_key', num_images=5)
        self.assertRaises(NotImplementedError, lambda: callback.on_cache('', {}))

        callback.on_cache = MagicMock()
        callback.on_cache.return_value = 'image'

        state = {'my_key': torch.ones(10, 3, 2, 2), torchbearer.BATCH: 0}
        callback.on_batch(state)
        self.assertTrue(callback.on_cache.call_args[0][0].sum() == 60)
        self.assertTrue(callback.on_cache.call_args[0][1] == state)
        callback.on_cache.reset_mock()
        callback.on_end_epoch({})
        callback.on_batch(state)
        self.assertTrue(callback.on_cache.call_args[0][0].sum() == 60)
        self.assertTrue(callback.on_cache.call_args[0][1] == state)


class TestMakeGrid(TestCase):
    @patch('torchvision.utils.make_grid')
    def test_main(self, mock_grid):
        mock_grid.return_value = 10

        callback = imaging.MakeGrid(key='x', num_images=18, nrow=9, padding=3, normalize=True, norm_range='tmp',
                                    scale_each=True, pad_value=1)

        res = callback.on_cache('test', {})
        mock_grid.assert_called_once_with('test', nrow=9, padding=3, normalize=True, range='tmp', scale_each=True,
                                          pad_value=1)
        self.assertTrue(res == 10)


class TestFromState(TestCase):
    def test_main(self):
        callback = imaging.FromState('test')

        self.assertTrue(callback.on_batch({torchbearer.EPOCH: 0, 'test': 1}) == 1)
        self.assertTrue(callback.on_batch({torchbearer.EPOCH: 1, 'testing': 1}) is None)
