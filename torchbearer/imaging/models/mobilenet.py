from torchvision.models.mobilenet import model_urls
from torchvision.models.mobilenet import MobileNetV2 as MobileNet
from torchvision.models.utils import load_state_dict_from_url
import torch.nn.functional as F
from torchbearer.imaging.models.utils import storer, basemodel


layer_names = ['conv1BnRelu1', *['invertedresidual{}'.format(i) for i in range(18)], 'avgpool18', 'dropout18', 'fc18']


class MobileNetV2(MobileNet):
    def forward(self, x, state):
        for i, m in enumerate(self.features):
            x = m(x)
            x = storer(state, layer_names[i], x)
        x = x.mean([2, 3])
        x = storer(state, layer_names[i+1], x)
        for j, m in enumerate(self.classifier):
            x = m(x)
            x = storer(state, layer_names[i+j+2], x)

        return x


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = basemodel(MobileNetV2)(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], progress=progress)
        model.load_state_dict(state_dict)
    return model
