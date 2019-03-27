import torch.nn as nn
import torchvision
from torchvision import transforms

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = torchvision.models.googlenet(True)

    def forward(self, input):
        if input is not None:
            return self.net(input)[2]


model = Model()

import torchbearer
from torchbearer import Trial
import torchbearer.callbacks.imaging as imaging

VIS = torchbearer.state_key('vis')

trial = Trial(model, callbacks=[
    imaging.ClassAppearanceModel(1000, (3, 224, 224), steps=100, target=951, verbose=-1, transform=inv_normalize).on_val().to_file('lemon.png'),
    imaging.ClassAppearanceModel(1000, (3, 224, 224), steps=10000, target=968, verbose=-1, transform=inv_normalize).on_val().to_file('cup.png')
])
trial.for_val_steps(1).to('cuda')
trial.evaluate()
