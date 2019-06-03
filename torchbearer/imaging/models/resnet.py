from torchvision.models.resnet import model_urls, BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet as BaseResNet
from torchvision.models.utils import load_state_dict_from_url

from torchbearer.imaging.models.utils import storer, basemodel

layer_names = ['conv1', 'batchnorm1', 'relu1', 'maxpool1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool5', 'fc6']


class ResNet(BaseResNet):
    def forward(self, x, state):
        x = self.conv1(x)
        x = storer(state, layer_names[0], x)
        x = self.bn1(x)
        x = storer(state, layer_names[1], x)
        x = self.relu(x)
        x = storer(state, layer_names[2], x)
        x = self.maxpool(x)
        x = storer(state, layer_names[3], x)

        x = self.layer1(x)
        x = storer(state, layer_names[4], x)
        x = self.layer2(x)
        x = storer(state, layer_names[5], x)
        x = self.layer3(x)
        x = storer(state, layer_names[6], x)
        x = self.layer4(x)
        x = storer(state, layer_names[7], x)

        x = self.avgpool(x)
        x = storer(state, layer_names[8], x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = storer(state, layer_names[9], x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = basemodel(ResNet)(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-50 32x4d model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8d model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)