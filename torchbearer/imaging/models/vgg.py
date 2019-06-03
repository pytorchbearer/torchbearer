from torchvision.models.vgg import make_layers, cfgs
from torchvision.models import VGG as VGGNet
from torchvision.models.vgg import model_urls
from torchvision.models.utils import load_state_dict_from_url

from torchbearer.imaging.models.utils import storer, basemodel


def _make_block_layer_names(layers, length):
    layer_names = []
    subindex = 'a'
    for i in range(length):
        for l in layers:
            if length == 1:
                layer_names.append(l)
            else:
                layer_names.append(l + subindex)
        subindex = chr(ord(subindex) + 1)
    return layer_names


def _make_vgg_layer_names(cfg, bn=False):
    if cfg == 'A':
        depths = [1, 1, 2, 2, 2]
    elif cfg == 'B':
        depths = [2, 2, 2, 2, 2]
    elif cfg == 'D':
        depths = [2, 2, 3, 3, 3]
    else:
        depths = [2, 2, 4, 4, 4]

    if bn:
        return [
                *_make_block_layer_names(['conv1', 'batchnorm1', 'relu1'], depths[0]), 'maxpool1',
                *_make_block_layer_names(['conv2', 'batchnorm2', 'relu2'], depths[1]), 'maxpool2',
                *_make_block_layer_names(['conv3', 'batchnorm3', 'relu3'], depths[2]), 'maxpool3',
                *_make_block_layer_names(['conv4', 'batchnorm4', 'relu4'], depths[3]), 'maxpool4',
                *_make_block_layer_names(['conv5', 'batchnorm5', 'relu5'], depths[4]), 'maxpool5',
                'avgpool6', 'fc7a', 'relu7a', 'dropout7a', 'fc7b', 'relu7b', 'dropout7b', 'fc7c']
    else:
        return [
                *_make_block_layer_names(['conv1', 'relu1'], depths[0]), 'maxpool1',
                *_make_block_layer_names(['conv2', 'relu2'], depths[1]), 'maxpool2',
                *_make_block_layer_names(['conv3', 'relu3'], depths[2]), 'maxpool3',
                *_make_block_layer_names(['conv4', 'relu4'], depths[3]), 'maxpool4',
                *_make_block_layer_names(['conv5', 'relu5'], depths[4]), 'maxpool5',
                'avgpool6', 'fc7a', 'relu7a', 'dropout7a', 'fc7b', 'relu7b', 'dropout7b', 'fc7c']


layer_names = {
    'vgg11': _make_vgg_layer_names('A'),
    'vgg11_bn': _make_vgg_layer_names('A', True),
    'vgg13': _make_vgg_layer_names('B'),
    'vgg13_bn': _make_vgg_layer_names('B', True),
    'vgg16': _make_vgg_layer_names('D'),
    'vgg16_bn': _make_vgg_layer_names('D', True),
    'vgg19': _make_vgg_layer_names('E'),
    'vgg19_bn': _make_vgg_layer_names('E', True),
}


class VGG(VGGNet):
    def __init__(self, features, arch, num_classes=1000, init_weights=True):
        super().__init__(features, num_classes, init_weights)
        self.arch = arch

    def forward(self, x, state):
        for i, m in enumerate(self.features):
            x = m(x)
            x = storer(state, layer_names[self.arch][i], x)
        i += 1
        x = self.avgpool(x)
        x = storer(state, layer_names[self.arch][i], x)
        x = x.view(x.size(0), -1)
        i += 1
        for j, m in enumerate(self.classifier):
            x = m(x)
            x = storer(state, layer_names[self.arch][i + j], x)
        return x


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = basemodel(VGG)(make_layers(cfgs[cfg], batch_norm=batch_norm), arch,  **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)


import torch
from torchbearer.imaging.models.utils import LAYER_DICT
v = vgg16_bn()
data = torch.rand(1, 3, 256, 256)
state = {}
v(data, state)
print(state[LAYER_DICT].keys())