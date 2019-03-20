import torch
import torch.nn as nn
import torch.optim as optim
import torchbearer
from . import imaging, once_per_epoch

_deep_inside = """
@article{simonyan2013deep,
  title={Deep inside convolutional networks: Visualising image classification models and saliency maps},
  author={Simonyan, Karen and Vedaldi, Andrea and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1312.6034},
  year={2013}
}
"""

RANDOM = -10
ORDERED = -20


@torchbearer.cite(_deep_inside)
class ClassAppearanceModel(imaging.ImagingCallback):
    """The :class:`.ClassAppearanceModel` callback implements Figure 1 from
    `Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps <https://arxiv.org/abs/1312.6034>`_

    """
    def __init__(self, nimages, nclasses, input_size,
                 optimizer_factory=lambda params: optim.Adam(params, lr = 0.05),
                 logit_key=torchbearer.PREDICTION, prob_key=None, target=RANDOM, steps=1024, transform=None):
        super(ClassAppearanceModel, self).__init__(transform=transform)

        self.nimages = nimages
        self.nclasses = nclasses
        self.input_size = input_size
        self.optimizer_factory = optimizer_factory
        self.logit_key = logit_key
        # self.prob_key = prob_key
        self.target = target
        self.steps = steps

        self._target_keys = []

    @torchbearer.fluent
    def target_to_key(self, key):
        self._target_keys.append(key)

    @once_per_epoch
    def on_batch(self, state):
        # input_image = torch.zeros(self.input_size, requires_grad=True,
        #                           device=top_state[torchbearer.DEVICE], dtype=top_state[torchbearer.DATA_TYPE])
        # args = [self.nimages]
        # for _ in self.input_size:
        #     args.append(1)
        # input_batch = nn.Parameter(input_image.unsqueeze(0).repeat(*args))

        # state = top_state.copy()
        training = state[torchbearer.MODEL].training
        state[torchbearer.MODEL].eval()

        targets = torch.randint(high=self.nclasses, size=(self.nimages, 1)).long().to(state[torchbearer.DEVICE])
        targets[0][0] = 251
        for key in self._target_keys:
            state[key] = targets
        targets_hot = torch.zeros(self.nimages, self.nclasses).to(state[torchbearer.DEVICE])
        targets_hot.scatter_(1, targets, 1)
        targets_hot = targets_hot.ge(0.5)

        def loss(state):
            img = state[torchbearer.MODEL].input_batch
            variation = (
                torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) +
                torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
            )

            # torch.masked_select(state[self.logit_key][2], targets_hot).exp().sum()
            # state[self.logit_key][0][:, 10, :, :].mean()

            return - 0.005 * torch.masked_select(state[self.logit_key][2], targets_hot).exp().sum() + 0.5 * img.abs().mean() + 0.05 * variation

        # base_model = state[torchbearer.MODEL].train()

        class Model(nn.Module):
            def __init__(self, input_size, nimages, base_model):
                super(Model, self).__init__()
                self.base_model = base_model
                for param in self.base_model.parameters():
                    param.requires_grad = False
                self.base_model.requires_grad = False
                input_image = torch.randn(input_size)
                args = [nimages]
                for _ in input_size:
                    args.append(1)
                # args[0] = nimages
                self.input_batch = nn.Parameter(input_image.unsqueeze(0).repeat(*args) + 0.5, requires_grad=True)

            def forward(self, _, state):
                try:
                    return self.base_model(self.input_batch, state)
                except TypeError:
                    return self.base_model(self.input_batch)

        model = Model(self.input_size, self.nimages, state[torchbearer.MODEL])
        trial = torchbearer.Trial(model, self.optimizer_factory(filter(lambda p: p.requires_grad, model.parameters())), loss, ['loss'], [])
        trial.for_train_steps(self.steps).to(state[torchbearer.DEVICE], state[torchbearer.DATA_TYPE])
        trial.run()
        print(targets)

        # for state[torchbearer.BATCH] in range(self.iterations):
        #     optimizer.zero_grad()
        #     try:
        #         state[torchbearer.Y_PRED] = state[torchbearer.MODEL](input_batch, state=state)
        #     except TypeError:
        #         state[torchbearer.Y_PRED] = state[torchbearer.MODEL](input_batch)
        #
        #     logits = state[self.logit_key]
        #     loss =

        if not training:
            state[torchbearer.MODEL].eval()

        return model.input_batch.squeeze(0)
