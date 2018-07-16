Quickstart Guide
====================================

Let's get using sconce. Here's a pre-trained ResNet18 from torchvision and some data from Cifar10:

.. code::

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar', train=True, download=True, transform=[transforms.ToTensor()]))
    trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=128, shuffle=True, num_workers=10)

    testset = torchvision.datasets.CIFAR10(root='./data/cifar', train=False,  download=True, transform=[transforms.ToTensor()]))
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=128, shuffle=False, num_workers=10)

    model = torchvision.models.resnet18(pretrained=True)

Typically we would need a training loop and a series of calls to backward, step etc.
Instead, with sconce, we can define our optimiser and some metrics ('acc' and 'loss' for simplicity) and let it do the work.