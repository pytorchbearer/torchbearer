from torchvision.models.squeezenet import model_urls
from torchvision.models.squeezenet import SqueezeNet as BaseSqueezeNet
from torchvision.models.utils import load_state_dict_from_url

from torchbearer.imaging.models.utils import storer, basemodel


layer_names = {
    '1_0': ['conv1', 'relu1', 'maxpool2', 'fire2a', 'fire2b', 'fire2c', 'maxpool3', 'fire3a', 'fire3b', 'fire3c',
            'fire3d', 'maxpool4', 'fire4a', 'dropout4', 'final_conv', 'relu4', 'avgpool'],
    '1_1': ['conv1', 'relu1', 'maxpool2', 'fire2a', 'fire2b', 'maxpool3', 'fire3a', 'fire3b', 'maxpool4', 'fire4a',
            'fire4b', 'fire4c', 'fire4d', 'dropout4', 'final_conv', 'relu4', 'avgpool']
}


class SqueezeNet(BaseSqueezeNet):
    def __init__(self, version='1_0', num_classes=1000):
        super().__init__(version, num_classes)
        self.version = version

    def forward(self, x, state):
        for i, m in enumerate(self.features):
            x = m(x)
            x = storer(state, layer_names[self.version][i], x)
        i += 1
        for j, m in enumerate(self.classifier):
            x = m(x)
            x = storer(state, layer_names[self.version][i+j], x)
        return x.view(x.size(0), -1)


def _squeezenet(version, pretrained, progress, **kwargs):
    model = basemodel(SqueezeNet)(version, **kwargs)
    if pretrained:
        arch = 'squeezenet' + version
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def squeezenet1_0(pretrained=False, progress=True, **kwargs):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_0', pretrained, progress, **kwargs)


def squeezenet1_1(pretrained=False, progress=True, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _squeezenet('1_1', pretrained, progress, **kwargs)



