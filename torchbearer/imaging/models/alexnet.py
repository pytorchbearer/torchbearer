from torchvision.models import AlexNet as ANet
from torchvision.models.alexnet import model_urls
from torchvision.models.utils import load_state_dict_from_url

from torchbearer.imaging.models.utils import storer, basemodel

layer_names = ['conv1', 'relu1', 'maxpool2', 'conv2', 'relu2', 'maxpool3', 'conv3a', 'relu3a', 'conv3b', 'relu3b',
               'conv3c', 'relu3c', 'maxpool4', 'avgpool4', 'dropout5a', 'fc5a', 'relu5a', 'dropout5b', 'fc5b', 'relu5b',
               'fc5c']


class AlexNet(ANet):
    def forward(self, x, state):
        for i, m in enumerate(self.features):
            x = m(x)
            x = storer(state, layer_names[i], x)
        x = self.avgpool(x)
        i += 1
        x = storer(state, layer_names[i], x)
        x = x.view(x.size(0), 256 * 6 * 6)
        i += 1

        for j, m in enumerate(self.classifier):
            x = m(x)
            x = storer(state, layer_names[i+j], x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = basemodel(AlexNet)(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
