import warnings

import torch
from torchvision.models import GoogLeNet as GNet
from torchvision.models.googlenet import model_urls, _GoogLeNetOuputs
from torchvision.models.utils import load_state_dict_from_url

from torchbearer.imaging.models.utils import storer, basemodel

layer_names = ['conv1', 'maxpool2', 'conv2', 'conv3', 'maxpool3', 'inception3a', 'inception3b', 'maxpool4',
               'inception4a', 'inception4b', 'inception4c', 'inception4d', 'inception4e', 'maxpool5', 'inception5a',
               'inception5b', 'avgpool6', 'dropout6', 'fc6', 'aux1', 'aux2'
               ]


class GoogLeNet(GNet):
    def forward(self, x, state):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        # N x 3 x 224 x 224
        x = self.conv1(x)
        x = storer(state, layer_names[0], x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        x = storer(state, layer_names[1], x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        x = storer(state, layer_names[2], x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        x = storer(state, layer_names[3], x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)
        x = storer(state, layer_names[4], x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        x = storer(state, layer_names[5], x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        x = storer(state, layer_names[6], x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        x = storer(state, layer_names[7], x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        x = storer(state, layer_names[8], x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
            x = storer(state, layer_names[19], x)

        x = self.inception4b(x)
        x = storer(state, layer_names[9], x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        x = storer(state, layer_names[10], x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        x = storer(state, layer_names[11], x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
            x = storer(state, layer_names[20], x)

        x = self.inception4e(x)
        x = storer(state, layer_names[12], x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        x = storer(state, layer_names[13], x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        x = storer(state, layer_names[14], x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        x = storer(state, layer_names[15], x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        x = storer(state, layer_names[16], x)
        # N x 1024 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 1024
        x = self.dropout(x)
        x = storer(state, layer_names[17], x)
        x = self.fc(x)
        x = storer(state, layer_names[18], x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:
            return _GoogLeNetOuputs(x, aux2, aux1)
        return x


def googlenet(pretrained=False, progress=True, **kwargs):
    r"""GoogLeNet (Inception v1) model architecture from
    `"Going Deeper with Convolutions" <http://arxiv.org/abs/1409.4842>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, adds two auxiliary branches that can improve training.
            Default: *False* when pretrained is True otherwise *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' not in kwargs:
            kwargs['aux_logits'] = False
        if kwargs['aux_logits']:
            warnings.warn('auxiliary heads in the pretrained googlenet model are NOT pretrained, '
                          'so make sure to train them')
        original_aux_logits = kwargs['aux_logits']
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False
        model = basemodel(GoogLeNet)(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['googlenet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.aux1, model.aux2
        return model

    return basemodel(GoogLeNet)(**kwargs)
