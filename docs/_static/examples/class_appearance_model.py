# import torch
import torch.nn as nn
# import torch.optim as optim
import torchvision
from torchvision import transforms

# from torchbearer.cv_utils import DatasetValidationSplitter

BATCH_SIZE = 128

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
# loss = nn.CrossEntropyLoss()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = torchvision.models.googlenet(True)
        # self.net = torchvision.models.alexnet(True)
        # self.net = torchvision.models.inception_v3(True)
        # self.net.aux_logits = False
        # self.net = torchvision.models.vgg16(True)
        # self.net = torchvision.models.resnet18(True)

    def forward(self, input):
        if input is not None:
            return self.net(input)


model = Model()

import torchbearer
from torchbearer import Trial
# from torchbearer.callbacks import TensorBoard

VIS = torchbearer.state_key('vis')

trial = Trial(model, callbacks=[torchbearer.callbacks.imaging.ClassAppearanceModel(1, 1000, (3, 224, 224), steps=1000000).on_train().to_state(VIS),
                                torchbearer.callbacks.imaging.FromState(key=VIS, transform=inv_normalize).on_train().to_file('vis2.png')])
trial.for_train_steps(1).to('cuda')
trial.run()

# torchbearer_trial = Trial(model, optimizer, loss, metrics=['acc', 'loss'], callbacks=[TensorBoard(write_batch_metrics=True)]).to('cuda')
# torchbearer_trial.with_generators(train_generator=traingen, val_generator=valgen, test_generator=testgen)
# torchbearer_trial.run(epochs=10)
# torchbearer_trial.evaluate(data_key=torchbearer.TEST_DATA)
