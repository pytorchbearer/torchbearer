from unittest import TestCase
import torch
from torchbearer.imaging.models.redirect_relu import RedirectedReLU, RedirectedReLUFunction, \
    RedirectedReLU6, RedirectedReLU6Function


class TestRedirectRelu(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu_examples = [
            (1., -1., 0.), (-1., -1., -1.),
            (1.,  1., 1.), (-1.,  1., -1.),
        ]
        self.relu6_examples = self.relu_examples + [(1.,  7., 1.), (-1.,  7.,  0.)]

    def test_redirected(self):
        for i in range(4):
            in_grad, in_tens, grad = self.relu_examples[i]
            in_grad, in_tens, grad = torch.Tensor([in_grad]).view(1,1), torch.Tensor([in_tens]).view(1,1), torch.Tensor([grad]).view(1,1)
            in_tens.requires_grad = True
            ctx = lambda: None
            out = RedirectedReLU.forward(ctx, in_tens)
            ctx.saved_tensors = in_tens
            pred_grad = RedirectedReLUFunction.backward(ctx, in_grad)
            self.assertTrue(pred_grad.item() == grad.item())

    def test_redirected6(self):
        for i in range(6):
            in_grad, in_tens, grad = self.relu6_examples[i]
            in_grad, in_tens, grad = torch.Tensor([in_grad]).view(1,1), torch.Tensor([in_tens]).view(1,1), torch.Tensor([grad]).view(1,1)
            in_tens.requires_grad = True
            ctx = lambda: None
            out = RedirectedReLU6.forward(ctx, in_tens)
            ctx.saved_tensors = in_tens
            pred_grad = RedirectedReLU6Function.backward(ctx, in_grad)
            self.assertTrue(pred_grad.item() == grad.item())
