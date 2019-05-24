import unittest
from mock import Mock, patch

import torchbearer.variational.datasets as ds


class TestMakeDataset(unittest.TestCase):
    @patch('os.walk')
    def test_make_dataset_all_files(self, mock_walk):
        mock_walk.return_value = [
            ('/root', (), ('b.a', 'c.a'))
        ]

        out = ds.make_dataset('a',['a'])
        self.assertTrue(out == ['/root/b.a', '/root/c.a'])

    @patch('os.walk')
    def test_make_dataset_some_files(self, mock_walk):
        mock_walk.return_value = [
            ('/root', (), ('b.v', 'c.a'))
        ]

        out = ds.make_dataset('a',['a'])
        self.assertTrue(out == ['/root/c.a'])

    @patch('os.walk')
    def test_make_dataset_no_files(self, mock_walk):
        mock_walk.return_value = [
            ('/root', (), ('b.v', 'c.v'))
        ]

        out = ds.make_dataset('a',['a'])
        self.assertTrue(out == [])

    @patch('os.walk')
    def test_make_dataset_no_root(self, mock_walk):
        mock_walk.return_value = [
            ('', (), ('b.a', 'c.a'))
        ]

        out = ds.make_dataset('a',['a'])
        self.assertTrue(out == ['b.a', 'c.a'])

    @patch('os.walk')
    def test_make_dataset_multiple_extensions(self, mock_walk):
        mock_walk.return_value = [
            ('', (), ('b.1', 'c.2'))
        ]

        out = ds.make_dataset('a',['1','2'])
        self.assertTrue(out == ['b.1', 'c.2'])


class TestSimpleImageFolder(unittest.TestCase):
    @patch.object(ds, 'make_dataset')
    def test_length(self, mock_md):
        loader = Mock()
        extensions = None
        transform = Mock()
        target_transform = Mock()

        images = ['a.1', 'b.1', 'c.1', 'd.1']
        mock_md.return_value = images
        dataset = ds.SimpleImageFolder('root', loader, extensions, transform, target_transform)

        self.assertTrue(len(dataset) == len(images))

    @patch.object(ds, 'make_dataset')
    def test_image_path(self, mock_md):
        loader = Mock()
        extensions = None
        transform = Mock()
        target_transform = Mock()

        images = ['a.1', 'b.1', 'c.1', 'd.1']
        mock_md.return_value = images
        dataset = ds.SimpleImageFolder('root', loader, extensions, transform, target_transform)

        _ = dataset[1]
        self.assertTrue(loader.call_args[0][0] == images[1])

        _ = dataset[2]
        self.assertTrue(loader.call_args[0][0] == images[2])

    @patch.object(ds, 'make_dataset')
    def test_transform_args(self, mock_md):
        loader = Mock()
        mock_image = Mock()
        loader.return_value = mock_image
        extensions = None
        transform = Mock()
        target_transform = Mock()

        images = ['a.1', 'b.1', 'c.1', 'd.1']
        mock_md.return_value = images
        dataset = ds.SimpleImageFolder('root', loader, extensions, transform, target_transform)

        _ = dataset[1]
        self.assertTrue(transform.call_args[0][0] == mock_image)

    @patch.object(ds, 'make_dataset')
    def test_target_transform_args(self, mock_md):
        loader = Mock()
        mock_image = Mock()
        loader.return_value = mock_image
        extensions = None
        transform = Mock()
        target_transform = Mock()

        images = ['a.1', 'b.1', 'c.1', 'd.1']
        mock_md.return_value = images
        dataset = ds.SimpleImageFolder('root', loader, extensions, transform, target_transform)

        _ = dataset[1]
        self.assertTrue(target_transform.call_args[0][0] == mock_image)


class TestCelebA(unittest.TestCase):
    @patch.object(ds.SimpleImageFolder, '__len__')
    @patch.object(ds.SimpleImageFolder, '__getitem__')
    def test_celeba(self, mock_gi, mock_len):
        return_val = 'test'
        mock_len.return_value = 10
        mock_gi.return_value = return_val

        celeba = ds.CelebA('root', Mock(), Mock())
        out = celeba[6]
        self.assertTrue(mock_gi.call_args[0][0] == 6)
        self.assertTrue(out == return_val)

    @patch.object(ds.SimpleImageFolder, '__len__')
    @patch.object(ds.SimpleImageFolder, '__getitem__')
    def test_celeba_hq_get_item(self, mock_gi, mock_len):
        return_val = 'test'
        mock_len.return_value = 10
        mock_gi.return_value = return_val

        celebahq = ds.CelebA_HQ('root', False, Mock())
        out = celebahq[6]

        self.assertTrue(mock_gi.call_args[0][0] == 6)
        self.assertTrue(out == return_val)

    @patch.object(ds.CelebA_HQ, 'npy_loader')
    @patch.object(ds.SimpleImageFolder, '__len__')
    @patch.object(ds.SimpleImageFolder, '__getitem__')
    def test_celeba_hq_use_numpy_loader(self, mock_gi, mock_len, mock_loader):
        mock_len.return_value = 10
        mock_gi.return_value = Mock()

        celebahq = ds.CelebA_HQ('root', True, Mock())
        out = celebahq[6]

        self.assertTrue(celebahq.loader == mock_loader)
        self.assertTrue(celebahq.extensions == ['npy'])

    @patch.object(ds.SimpleImageFolder, '__len__')
    @patch.object(ds.SimpleImageFolder, '__getitem__')
    @patch('torchvision.datasets.folder.default_loader')
    def test_celeba_hq_use_default_loader(self, mock_loader, mock_gi, mock_len, ):
        import torchvision
        mock_len.return_value = 10

        celebahq = ds.CelebA_HQ('root', False, Mock())
        out = celebahq[6]

        self.assertTrue(celebahq.loader == mock_loader)
        self.assertTrue(celebahq.extensions == torchvision.datasets.folder.IMG_EXTENSIONS)

    @patch.object(ds.SimpleImageFolder, '__len__')
    @patch('numpy.load')
    @patch('PIL.Image.fromarray')
    def test_celeba_hq_numpy_loader(self, mock_fromarray, mock_load, mock_len):
        samples = ['a.npy', 'b.npy', 'c.npy']
        mock_len.return_value = 3
        mock_output = Mock()
        mock_output.transpose.return_value = mock_output
        mock_load.return_value = (mock_output, )

        celebahq = ds.CelebA_HQ('root', True, Mock())
        celebahq.samples = samples
        out = celebahq[1]

        self.assertTrue(mock_load.call_args[0][0] == samples[1])
        self.assertTrue(mock_fromarray.call_args[0][0] == mock_output)


class TestdSprites(unittest.TestCase):
    @patch.object(ds.dSprites, 'download')
    @patch.object(ds.dSprites, 'load_data')
    @patch('numpy.load')
    def test_no_download(self, mock_load, mock_load_data, mock_download):
        sprites = ds.dSprites('root', False, None)
        self.assertTrue(sprites.download.call_count == 0)

    @patch.object(ds.dSprites, 'download')
    @patch.object(ds.dSprites, 'load_data')
    @patch('numpy.load')
    def test_do_download(self, mock_load, mock_load_data, mock_download):
        sprites = ds.dSprites('root', True, None)
        self.assertTrue(sprites.download.call_count == 1)

    @patch.object(ds.dSprites, 'download')
    @patch.object(ds.dSprites, 'load_data')
    @patch('numpy.load')
    def test_latents_bases(self, mock_load, mock_load_data, mock_download):
        sprites = ds.dSprites('root', False, None)
        self.assertTrue(list(sprites.latents_bases) == [737280, 245760, 40960, 1024, 32, 1])

    @patch.object(ds.dSprites, 'download')
    @patch('numpy.load')
    def test_load_data(self, mock_load, mock_download):
        sprites = ds.dSprites('root', False, None)
        self.assertTrue(mock_load.call_args_list[0][0][0] == 'root/imgs.npy')
        self.assertTrue(mock_load.call_args_list[1][0][0] == 'root/latents_values.npy')
        self.assertTrue(mock_load.call_args_list[2][0][0] == 'root/latents_classes.npy')

    @patch.object(ds.dSprites, 'download')
    @patch.object(ds.dSprites, '__getitem__')
    @patch('numpy.load')
    def test_image_by_latent(self, mock_load, mock_gi, mock_download):
        import numpy as np
        mock_gi.return_value = (None, )

        bases = np.array([737280, 245760, 40960, 1024, 32, 1])
        latent = [1,2,3,4,5,6]
        img_id = (bases * latent).sum()

        sprites = ds.dSprites('root', False, None)
        sprites.get_img_by_latent(latent)
        self.assertTrue(sprites.__getitem__.call_args[0][0] == img_id)

    @patch.object(ds.dSprites, 'download')
    @patch.object(ds.dSprites, 'load_data')
    @patch('numpy.load')
    def test_len(self, mock_load, mock_load_data, mock_download):
        import numpy as np
        mock_load_data.return_value = np.zeros((10, 5, 5))
        sprites = ds.dSprites('root', False, None)

        self.assertTrue(len(sprites) == 10)

    @patch('PIL.Image.fromarray')
    @patch.object(ds.dSprites, 'load_data')
    @patch('numpy.load')
    def test_get_item(self, mock_load, mock_load_data, mock_fromarray):
        import torch
        sprites = ds.dSprites('root', False, None)
        mock_data = torch.rand(5, 1)
        mock_data[1] *= 3
        sprites.data = mock_data
        out = sprites[1]

        self.assertTrue(mock_fromarray.call_args[0][0] == mock_data[1]*255)
        self.assertTrue(mock_fromarray.call_args[1] == {'mode':'L'})
        self.assertTrue(len(out) == 2)

    @patch('PIL.Image.fromarray')
    @patch.object(ds.dSprites, 'load_data')
    @patch('numpy.load')
    def test_get_item_transform(self, mock_load, mock_load_data, mock_fromarray):
        import torch
        transform = Mock()
        sprites = ds.dSprites('root', False, transform)
        mock_fromarray.return_value = 'test'
        mock_data = torch.rand(5, 1)
        mock_data[1] *= 3
        sprites.data = mock_data
        _ = sprites[1]

        self.assertTrue(transform.call_args[0][0] == 'test')

    @patch('zipfile.ZipFile')
    @patch('shutil.copyfileobj')
    @patch('os.makedirs')
    @patch.object(ds.dSprites, 'load_data')
    @patch('numpy.load')
    def test_download(self, mock_load, mock_load_data, mock_mkdirs, mock_copyfileobj, mock_zip, mock_open, mock_urlopen):
        mock_return = Mock()
        # mock_urlopen = Mock()
        # mock_url.return_value = mock_urlopen
        mock_urlopen.return_value.__enter__.return_value = mock_return

        root = 'root'
        filename = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

        sprites = ds.dSprites(root, True, None)

        self.assertTrue(mock_mkdirs.call_args[0][0] == "root")
        self.assertTrue(mock_mkdirs.call_args[1] == {'exist_ok': True})

        self.assertTrue(mock_urlopen.call_args[0][0] == 'https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true')
        self.assertTrue(mock_open.call_args[0][0] == 'root/' + filename)

        self.assertTrue(mock_copyfileobj.call_args[0][0] == mock_return)

    import sys
    if sys.version_info[0] < 3:
        test_download = patch('urllib2.urlopen')(patch('__builtin__.open')(test_download))
    else:
        test_download = patch('urllib.request.urlopen')(patch('builtins.open')(test_download))