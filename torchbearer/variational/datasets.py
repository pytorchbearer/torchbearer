import os
import shutil
import zipfile

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def make_dataset(dir, extensions):
    from torchvision.datasets.folder import has_file_allowed_extension
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images


class SimpleImageFolder(Dataset):
    def __init__(self, root, loader=None, extensions=None, transform=None, target_transform=None):
        """
        Simple image folder dataset that loads all images from inside a folder and returns items in (image, image) tuple

        Args:
            root (str): Root directory of dataset containing all aligned images
            loader (function, optional): Image loader function that takes a file or path and returns the loaded image (see torchvision.datasets.folder)
            extensions (:obj:`list` of :obj:`str`, optional): List of file extensions that can be loaded
            transform (``Transform``, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (``Transform``, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        """
        from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
        loader = default_loader if loader is None else loader
        extensions = IMG_EXTENSIONS if extensions is None else extensions

        samples = make_dataset(root, extensions)

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of image

        Returns:
            tuple: (sample, target) where target is target transformed image.
        """
        path = self.samples[index]
        sample = self.loader(path)
        input_sample, target_sample = sample, sample
        if self.transform is not None:
            input_sample = self.transform(sample)
        if self.target_transform is not None:
            target_sample = self.target_transform(sample)

        return input_sample, target_sample

    def __len__(self):
        return len(self.samples)


class CelebA(SimpleImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        """
        `CelebA <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ auto-encoding dataset

        Args:
            root (str): Root directory of dataset containing all aligned images in 'root'
            transform (``Transform``, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (``Transform``, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        """
        super(CelebA, self).__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        item = super(CelebA, self).__getitem__(index)
        return item


class CelebA_HQ(SimpleImageFolder):
    def __init__(self, root, as_npy=False, transform=None):
        """
        CelebA_HQ, high quality version of `celebA <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ auto-encoding dataset as introduced by `Progressive GAN <https://arxiv.org/abs/1710.10196>`_

        Args:
            root (str): Root directory of dataset containing all hq images in 'root'
            as_npy (bool, optional): If True, assume images are stored in numpy arrays. Else assume a standard image format
            transform (``Transform``, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        """
        from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
        if as_npy:
            loader = self.npy_loader
            extensions = ['npy']
        else:
            loader = default_loader
            extensions = IMG_EXTENSIONS
        super(CelebA_HQ, self).__init__(root, loader, extensions, transform)

    @staticmethod
    def npy_loader(path):
        img = np.load(path)[0].transpose([1,2,0])
        pil_image = Image.fromarray(img)
        return pil_image

    def __getitem__(self, index):
        item = super(CelebA_HQ, self).__getitem__(index)
        return item


class dSprites(Dataset):
    def __init__(self, root, download=False, transform=None):
        """
        `dSprites <https://github.com/deepmind/dsprites-dataset>`_ Dataset

        Args:
            root (str): Root directory of dataset containing 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz' or to download it to
            download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again.
            transform (``Transform``, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        """
        super(dSprites, self).__init__()
        self.file = root
        self.transform = transform

        if download:
            self.download()

        self.data = self.load_data()
        self.latents_sizes = np.array([1, 3, 6, 40, 32, 32])
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1, ])))
        self.latents_values = np.load(os.path.join(self.file, "latents_values.npy"))
        self.latents_classes = np.load(os.path.join(self.file, "latents_classes.npy"))

    def download(self):
        if not os.path.exists(os.path.join(self.file, "imgs.npy")):
            data_url = 'https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true'
            import sys
            if sys.version_info[0] < 3:
                import urllib2 as request
            else:
                import urllib.request as request
            file = os.path.join(self.file, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

            os.makedirs(self.file, exist_ok=True)
            with request.urlopen(data_url) as response, open(file, 'wb+') as out_file:
                shutil.copyfileobj(response, out_file)

            zip_ref = zipfile.ZipFile(file, 'r')
            zip_ref.extractall(self.file)
            zip_ref.close()

    def get_img_by_latent(self, latent_code):
        """
        Returns the image defined by the latent code
        
        Args:
            latent_code (:obj:`list` of :obj:`int`): Latent code of length 6 defining each generative factor

        Returns:
            Image defined by given code

        """
        def latent_to_index(latents):
            return np.dot(latents, self.latents_bases).astype(int)
        idx = latent_to_index(latent_code)
        return self.__getitem__(idx)[0]

    def load_data(self):
        root = os.path.join(self.file, "imgs.npy")
        data = np.load(root)
        return data

    def __getitem__(self, index):
        data = self.data[index]
        data = Image.fromarray(data * 255, mode='L')

        if self.transform is not None:
            data = self.transform(data)

        return data, data

    def __len__(self):
        return self.data.shape[0]
