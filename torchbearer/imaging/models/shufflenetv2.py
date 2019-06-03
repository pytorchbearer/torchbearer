from torchvision.models.shufflenetv2 import model_urls
from torchvision.models.shufflenetv2 import ShuffleNetV2 as ShuffleNet
from torchvision.models.utils import load_state_dict_from_url

from torchbearer.imaging.models.utils import storer, basemodel


layer_names = ['conv1', 'maxpool2', 'stage2', 'stage3', 'stage4', 'conv5', 'avgpool6', 'fc6']


class ShuffleNetV2(ShuffleNet):
    def forward(self, x, state):
        x = self.conv1(x)
        x = storer(state, layer_names[0], x)
        x = self.maxpool(x)
        x = storer(state, layer_names[1], x)
        x = self.stage2(x)
        x = storer(state, layer_names[2], x)
        x = self.stage3(x)
        x = storer(state, layer_names[3], x)
        x = self.stage4(x)
        x = storer(state, layer_names[4], x)
        x = self.conv5(x)
        x = storer(state, layer_names[5], x)
        x = x.mean([2, 3])  # globalpool
        x = storer(state, layer_names[6], x)
        x = self.fc(x)
        x = storer(state, layer_names[7], x)
        return x


def _shufflenetv2(arch, pretrained, progress, *args, **kwargs):
    model = basemodel(ShuffleNetV2)(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict)

    return model


def shufflenet_v2_x0_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, progress,
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.0', pretrained, progress,
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.5', pretrained, progress,
                         [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x2.0', pretrained, progress,
                         [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)