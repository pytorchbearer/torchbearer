from torchvision.models.inception import model_urls, _InceptionOuputs
from torchvision.models.inception import Inception3 as Inception3Net
from torchvision.models.utils import load_state_dict_from_url
import torch
import torch.nn.functional as F
from torchbearer.imaging.models.utils import storer, basemodel


layer_names = ['conv1a', 'conv2a', 'conv2b', 'maxpool3', 'conv3b', 'conv4a', 'maxpool5', 'mixed5b', 'mixed5c',
               'mixed5d', 'mixed6a', 'mixed6b', 'mixed6c', 'mixed6d', 'mixed6e', 'mixed7a', 'mixed7b', 'mixed7c',
               'avgpool8', 'dropout8b', 'fc8c', 'aux']


class Inception3(Inception3Net):
    def forward(self, x, state):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        x = storer(state, layer_names[0], x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        x = storer(state, layer_names[1], x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        x = storer(state, layer_names[2], x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = storer(state, layer_names[3], x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        x = storer(state, layer_names[4], x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        x = storer(state, layer_names[5], x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = storer(state, layer_names[6], x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        x = storer(state, layer_names[7], x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        x = storer(state, layer_names[8], x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        x = storer(state, layer_names[9], x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        x = storer(state, layer_names[10], x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        x = storer(state, layer_names[11], x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        x = storer(state, layer_names[12], x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        x = storer(state, layer_names[13], x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        x = storer(state, layer_names[14], x)
        # N x 768 x 17 x 17
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
            x = storer(state, layer_names[21], x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        x = storer(state, layer_names[15], x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        x = storer(state, layer_names[16], x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        x = storer(state, layer_names[17], x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = storer(state, layer_names[18], x)
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        x = storer(state, layer_names[19], x)
        # N x 2048 x 1 x 1
        x = x.view(x.size(0), -1)
        # N x 2048
        x = self.fc(x)
        x = storer(state, layer_names[20], x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:
            return _InceptionOuputs(x, aux)
        return x


def inception_v3(pretrained=False, progress=True, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' in kwargs:
            original_aux_logits = kwargs['aux_logits']
            kwargs['aux_logits'] = True
        else:
            original_aux_logits = True
        model = basemodel(Inception3)(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['inception_v3_google'], progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.AuxLogits
        return model

    return basemodel(Inception3)(**kwargs)
