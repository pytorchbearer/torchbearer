import unittest
from mock import patch, Mock

import torch

import torchbearer
import torchbearer.variational.visualisation as vis


class SimpleLatentWalker(vis.LatentWalker):
    def vis(self, state):
        return None


class TestLatentWalker(unittest.TestCase):
    @patch('torchbearer.callbacks.once_per_epoch')
    def test_on_train_on_val(self, mock_ope):
        lw = SimpleLatentWalker(False, 8)
        lw.dev = 'cpu'
        lw.on_train()
        lw.on_val()

        self.assertTrue(mock_ope.call_count == 2)
        self.assertTrue(mock_ope.call_args[0][0] == lw._vis)

    def test_for_space(self):
        space_id = 3
        lw = SimpleLatentWalker(False, 8)
        lw.dev = 'cpu'
        lw.for_space(space_id)

        self.assertTrue(lw.variable_space == space_id)

    def test_data_key(self):
        data_key = 3
        lw = SimpleLatentWalker(False, 8)
        lw.dev = 'cpu'
        lw.for_data(data_key)

        self.assertTrue(lw.data_key == data_key)

    def test_to_key_to_file(self):
        out_key = 'test_key'
        out_file = 'test_file'
        lw = SimpleLatentWalker(False, 8)
        lw.dev = 'cpu'
        lw.to_key(out_key)
        lw.to_file(out_file)

        self.assertTrue(lw.file == out_file)
        self.assertTrue(lw.store_key == out_key)

    def test_vis(self):
        state = {torchbearer.MODEL: Mock(), torchbearer.X: Mock(), torchbearer.DEVICE: 'cpu'}

        lw = SimpleLatentWalker(False, 8)
        lw.dev = 'cpu'
        lw.vis = Mock()
        lw._vis(state)

        self.assertTrue(lw.vis.call_count == 1)
        self.assertTrue(lw.vis.call_args[0][0] == state)

    def test_vis_to_file(self):
        state = {torchbearer.MODEL: Mock(), torchbearer.X: Mock(), torchbearer.DEVICE: 'cpu'}

        lw = SimpleLatentWalker(False, 8)
        lw.to_file('test_file')
        lw.dev = 'cpu'
        lw.vis = Mock()
        lw.vis.return_value = "Test"
        lw._save_walk = Mock()
        lw._vis(state)

        self.assertTrue(lw._save_walk.call_count == 1)
        self.assertTrue(lw._save_walk.call_args[0][0] == "Test")

    def test_vis_to_key(self):
        state = {torchbearer.MODEL: Mock(), torchbearer.X: Mock(), torchbearer.DEVICE: 'cpu'}

        lw = SimpleLatentWalker(False, 8)
        lw.to_key('test_key')
        lw.dev = 'cpu'
        lw.vis = Mock()
        lw.vis.return_value = "Test"
        lw._vis(state)

        self.assertTrue(state['test_key'] == "Test")

    def test_data_key_str(self):
        data_key = 'test_key'
        state = {torchbearer.MODEL: Mock(), torchbearer.X: Mock(), torchbearer.DEVICE: 'cpu', data_key: 3}

        lw = SimpleLatentWalker(False, 8)
        lw.for_data(data_key)
        lw.dev = 'cpu'
        lw.vis = Mock()
        lw.vis.return_value = "Test"
        lw._vis(state)

        self.assertTrue(lw.data == state['test_key'])

    @patch('torchvision.utils.save_image')
    def test_save_walk(self, mock_save_img):
        state = {torchbearer.MODEL: Mock(), torchbearer.X: Mock(), torchbearer.DEVICE: 'cpu'}

        lw = SimpleLatentWalker(False, 8)
        lw.to_file('test_file')
        lw.dev = 'cpu'
        lw.vis = Mock()
        lw.vis.return_value = "Test"
        lw._vis(state)

        self.assertTrue(mock_save_img.call_args[0][0] == "Test")
        self.assertTrue(mock_save_img.call_args[0][1] == "test_file")

    def test_empty_method(self):
        walker = vis.LatentWalker(False, 0)
        self.assertRaises(NotImplementedError, lambda: walker.vis({}))


class TestReconstructionViewer(unittest.TestCase):
    def test_recon(self):
        import torch
        state = {torchbearer.X: torch.rand(4), torchbearer.Y_PRED: torch.rand(4)}
        row_size = 2

        rv = vis.ReconstructionViewer(row_size=row_size)
        rv.data = state[torchbearer.X]
        rv.dev = 'cpu'
        out = rv.vis(state)

        self.assertTrue((out == torch.cat([state[torchbearer.X][:row_size], state[torchbearer.Y_PRED][:row_size]])).all())


class TestLinspaceWalker(unittest.TestCase):
    def __init__(self, methodName):
        super(TestLinspaceWalker, self).__init__(methodName)
        import torch
        self.state = {torchbearer.X: torch.rand(12,1,2,2), torchbearer.Y_PRED: torch.rand(2)}

        self.codes = torch.rand(12,2)
        self.outputs = torch.rand(12,1,2,2)

        self.mock_model = Mock()
        self.mock_model.latent_dims = [2,]
        self.mock_model.encode.return_value = [self.codes]
        self.mock_model.decode.return_value = self.outputs

    def test_single_dim(self):
        lw = vis.LinSpaceWalker(dims_to_walk=[1], lin_steps=3)
        lw.data = self.state[torchbearer.X]
        lw.model = self.mock_model
        lw.dev = 'cpu'
        lw.vis(self.state)

        correct_lin_sample = self.codes[0].repeat(3, 1)
        correct_lin_sample[0,1], correct_lin_sample[1,1], correct_lin_sample[2,1] = -1.0, 0.0, 1.0
        self.assertTrue((self.mock_model.decode.call_args[0][0][0] - correct_lin_sample[:3] < 1e-5).all())

    def test_single_dim_alt(self):
        lw = vis.LinSpaceWalker(dims_to_walk=[0], lin_steps=3)
        lw.data = self.state[torchbearer.X]
        lw.model = self.mock_model
        lw.dev = 'cpu'
        lw.vis(self.state)

        correct_lin_sample = self.codes[0].repeat(3, 1)
        correct_lin_sample[0,0], correct_lin_sample[1,0], correct_lin_sample[2,0] = -1.0, 0.0, 1.0
        self.assertTrue((self.mock_model.decode.call_args[0][0][0] - correct_lin_sample[:3] < 1e-5).all())

    def test_single_dim_zero_init(self):
        lw = vis.LinSpaceWalker(dims_to_walk=[1], lin_steps=3, zero_init=True)
        lw.data = self.state[torchbearer.X]
        lw.model = self.mock_model
        lw.dev = 'cpu'
        lw.vis(self.state)

        correct_lin_sample = torch.zeros(self.codes[0].shape).repeat(3, 1)
        correct_lin_sample[0,1], correct_lin_sample[1,1], correct_lin_sample[2,1] = -1.0, 0.0, 1.0
        self.assertTrue((self.mock_model.decode.call_args[0][0][0] - correct_lin_sample < 1e-5).all())

    def test_multi_dim(self):
        lw = vis.LinSpaceWalker(dims_to_walk=[0, 1], lin_steps=3)
        lw.data = self.state[torchbearer.X]
        lw.model = self.mock_model
        lw.dev = 'cpu'
        lw.vis(self.state)

        correct_lin_sample = torch.cat([self.codes[0].repeat(3, 1), self.codes[1].repeat(3, 1)])
        correct_lin_sample[0,0], correct_lin_sample[1,0], correct_lin_sample[2,0] = -1.0, 0.0, 1.0
        correct_lin_sample[3,1], correct_lin_sample[4,1], correct_lin_sample[5,1] = -1.0, 0.0, 1.0

        self.assertTrue((self.mock_model.decode.call_args[0][0][0] - correct_lin_sample < 1e-5).all())

    def test_multi_dim_same_image(self):
        lw = vis.LinSpaceWalker(dims_to_walk=[0, 1], lin_steps=3, same_image=True)
        lw.data = self.state[torchbearer.X]
        lw.model = self.mock_model
        lw.dev = 'cpu'
        lw.vis(self.state)

        correct_lin_sample = self.codes[0].repeat(6, 1)
        correct_lin_sample[0,0], correct_lin_sample[1,0], correct_lin_sample[2,0] = -1.0, 0.0, 1.0
        correct_lin_sample[3,1], correct_lin_sample[4,1], correct_lin_sample[5,1] = -1.0, 0.0, 1.0

        self.assertTrue((self.mock_model.decode.call_args[0][0][0] - correct_lin_sample < 1e-5).all())

    def test_multi_dim_zero_init(self):
        lw = vis.LinSpaceWalker(dims_to_walk=[0, 1], lin_steps=3, zero_init=True)
        lw.data = self.state[torchbearer.X]
        lw.model = self.mock_model
        lw.dev = 'cpu'
        lw.vis(self.state)

        correct_lin_sample = torch.cat([torch.zeros(self.codes[0].shape).repeat(3, 1), torch.zeros(self.codes[1].shape).repeat(3, 1)])
        correct_lin_sample[0,0], correct_lin_sample[1,0], correct_lin_sample[2,0] = -1.0, 0.0, 1.0
        correct_lin_sample[3,1], correct_lin_sample[4,1], correct_lin_sample[5,1] = -1.0, 0.0, 1.0

        self.assertTrue((self.mock_model.decode.call_args[0][0][0] - correct_lin_sample < 1e-5).all())

    def test_limits(self):
        lw = vis.LinSpaceWalker(dims_to_walk=[0, 1], lin_steps=3, lin_start=-2, same_image=True)
        lw.data = self.state[torchbearer.X]
        lw.model = self.mock_model
        lw.dev = 'cpu'
        lw.vis(self.state)

        correct_lin_sample = self.codes[0].repeat(6, 1)
        correct_lin_sample[0,0], correct_lin_sample[1,0], correct_lin_sample[2,0] = -2.0, -0.5, 1.0
        correct_lin_sample[3,1], correct_lin_sample[4,1], correct_lin_sample[5,1] = -2.0, -0.5, 1.0

        self.assertTrue((self.mock_model.decode.call_args[0][0][0] - correct_lin_sample < 1e-5).all())


class TestRandomWalker(unittest.TestCase):
    def __init__(self, methodName):
        super(TestRandomWalker, self).__init__(methodName)
        import torch
        self.state = {torchbearer.X: torch.rand(12,1,2,2), torchbearer.Y_PRED: torch.rand(2)}

        self.codes = torch.rand(12,2)
        self.outputs = torch.rand(32,1,2,2)

        self.mock_model = Mock()
        self.mock_model.latent_dims = [2,]
        self.mock_model.encode.return_value = [self.codes]
        self.mock_model.decode.return_value = self.outputs

    def test_code_shape(self):
        rw = vis.RandomWalker()
        rw.data = self.state[torchbearer.X]
        rw.model = self.mock_model
        rw.dev = 'cpu'
        rw.vis(self.state)

        self.assertTrue(list(self.mock_model.decode.call_args[0][0][0].shape) == [32, 2])

    def test_code_shape_alt(self):
        self.mock_model.decode.return_value = torch.rand(10, 1, 2, 2)

        rw = vis.RandomWalker(num_images=10)
        rw.data = self.state[torchbearer.X]
        rw.model = self.mock_model
        rw.dev = 'cpu'
        rw.vis(self.state)

        self.assertTrue(list(self.mock_model.decode.call_args[0][0][0].shape) == [10, 2])

    @patch('torch.randn')
    @patch('torch.rand')
    def test_uniform(self, mock_rand, mock_randn):
        mock_rand.return_value = torch.ones(2,2)
        mock_randn.return_value = torch.zeros(2,2)

        rw = vis.RandomWalker(num_images=10)
        rw.data = self.state[torchbearer.X]
        rw.model = self.mock_model
        rw.dev = 'cpu'
        rw.vis(self.state)

        self.assertTrue(mock_rand.call_count == 0)
        self.assertTrue(mock_randn.call_count == 1)

        rw = vis.RandomWalker(num_images=10, uniform=True)
        rw.data = self.state[torchbearer.X]
        rw.dev = 'cpu'
        rw.model = self.mock_model
        rw.vis(self.state)
        self.assertTrue(mock_rand.call_count == 1)
        self.assertTrue(mock_randn.call_count == 1)


class TestCodePathWalker(unittest.TestCase):
    def __init__(self, methodName):
        super(TestCodePathWalker, self).__init__(methodName)
        import torch
        self.state = {torchbearer.X: torch.rand(12,1,2,2), torchbearer.Y_PRED: torch.rand(2)}

        self.codes = torch.rand(12,2)
        self.outputs = torch.rand(2*5,1,2,2)

        self.mock_model = Mock()
        self.mock_model.latent_dims = [2,]
        self.mock_model.encode.return_value = [self.codes]
        self.mock_model.decode.return_value = self.outputs

    def test_vis(self):
        code1 = torch.eye(2, 2)
        code2 = (torch.eye(2, 2) == 0).to(torch.float)

        pw = vis.CodePathWalker(5, code1, code2)
        pw.data = self.state[torchbearer.X]
        pw.model = self.mock_model
        pw.dev = 'cpu'
        pw.vis(self.state)

        correct_code = torch.zeros(10, 2)
        line = torch.Tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        line_inv = torch.Tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        correct_code[:5, 0], correct_code[5:, 0] = line_inv, line
        correct_code[:5, 1], correct_code[5:, 1] = line, line_inv

        self.assertTrue((self.mock_model.decode.call_args[0][0] == correct_code).all())


class TestImagePathWalker(unittest.TestCase):
    def __init__(self, methodName):
        super(TestImagePathWalker, self).__init__(methodName)
        import torch
        self.state = {torchbearer.X: torch.rand(12,1,2,2), torchbearer.Y_PRED: torch.rand(2)}

        self.codes = torch.rand(12,2)
        self.outputs = torch.rand(2*5,1,2,2)

        self.mock_model = Mock()
        self.mock_model.latent_dims = [2,]
        self.mock_model.encode.return_value = [self.codes]
        self.mock_model.decode.return_value = self.outputs

    @patch('torchbearer.variational.CodePathWalker.vis')
    def test_vis(self, _):
        im1 = torch.rand(2, 5, 5)
        im2 = torch.rand(2, 5, 5)
        ipw = vis.ImagePathWalker(4, im1, im2)
        ipw.data = self.state[torchbearer.X]
        ipw.model = self.mock_model
        ipw.dev = 'cpu'
        ipw.vis(self.state)

        self.assertTrue((self.mock_model.encode.call_args_list[0][0][0] == im1).all())
        self.assertTrue((self.mock_model.encode.call_args_list[1][0][0] == im2).all())
