import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchbearer.imaging.models.alexnet import alexnet

# inv_normalize = transforms.Normalize(
#     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#     std=[1/0.229, 1/0.224, 1/0.225]
# )

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


model = alexnet(True)
print(model.get_layer_names())

import torchbearer
from torchbearer import Trial
import torchbearer.imaging as imaging
from torchbearer.imaging import Image as my_image, FFTImage, TensorImage
import torchbearer.imaging.transforms as my_transforms
import torch.optim as optim

transform = my_transforms.Compose([
    # lambda img: torch.cat((normalize(img[:3]), img[3].unsqueeze(0)), dim=0),
    my_transforms.RandomAlpha(sd=0.2, colour=True, decay_power=1.5),
    my_transforms.SpatialJitter(2),
    my_transforms.RandomScale([1], mode='nearest'),
    my_transforms.RandomScale([0.6, 0.7, 0.8, 0.9, 1.0, 1.1]),
    my_transforms.SpatialJitter(150),
    lambda img: img.pow(0.7),  # Gamma correction
    normalize,
    # my_transforms.RandomRotate(list(range(-10,10)) + list(range(-5,5)) + 10*list(range(-2,2))),
    # my_transforms.SpatialJitter(2)
])

# # loss = imaging.DeepDream(torchbearer.PREDICTION)
# loss = imaging.Channel(100, torchbearer.PREDICTION)
# # loss -= 5e-4 * imaging.TotalVariation()
# # loss -= 1e-3 * imaging.L1(constant=0, channels=lambda x: x[:3])
# # loss -= 0.05 * imaging.L2(constant=0.5, channels=lambda x: x[:3])
# loss -= 1e-4 * imaging.L1(constant=0, channels=lambda x: x[-1].unsqueeze(0))
# # loss -= 0.5 * imaging.Blur(channels=lambda x: x[:3])
# loss -= 0.1 * imaging.BlurAlpha()

#
# from PIL import Image
# input_image2 = Image.open('image3.jpg')
# input_image2 = transforms.ToTensor()(transforms.Resize(224)(input_image2))
# sz = input_image2.size()
# input_image2 = my_image.from_tensor(input_image2, transform=transform, sigmoid=False, decorrelate=False, requires_grad=False)

# input_image = TensorImage(torch.randn(3, 256, 256) * 0.01, transform=transform, decorrelate=False).sigmoid()
input_image = FFTImage((4, 512, 512), transform=transform, correlate=True).sigmoid()
# input_image = my_image.from_tensor(torch.randn(3, 512, 512) * 0.01, transform=transform, decorrelate=True).sigmoid()
# input_image += input_image2

optimizer = optim.Adam(input_image.parameters(), lr=0.03)


@imaging.criterion
def alpha_loss(state):
    alpha = input_image.get_valid_image()[-1].mean()
    # alpha_crop = state[torchbearer.INPUT][0, -1].mean()
    obj = state[torchbearer.PREDICTION][0, 366].mean()  # 340].mean()  # [0, 366].mean()  # 385].mean()
    return obj * (1.0 - alpha)  #  * 0.5) - (obj * (1.0 - alpha_crop))


loss = alpha_loss  # + 1e-4 * imaging.L1(constant=0, channels=lambda x: x[-1].unsqueeze(0))
# loss -= 0.001 * imaging.Blur(channels=lambda x: x[:3])

imaging.BasicAscent(input_image, loss, optimizer=optimizer, steps=500,
                    transform=my_transforms.Compose([
                        # lambda img: torch.cat((img[:3] ** 2.2, img[3].unsqueeze(0)), dim=0),
                        # lambda img: img[:3] * img[3].unsqueeze(0).repeat(3, 1, 1),
                        # my_transforms.RandomAlpha(sd=0.2, colour=True, decay_power=1.5),
                        # lambda img: img ** 2.2
                        lambda img: torch.cat((img[:3].pow(0.7), img[3].unsqueeze(0)), dim=0),  # Gamma correction
                    ])).to_file('test.png').run(model, device='cuda')

# trial = Trial(model, callbacks=[
#     imaging.ClassAppearanceModel(1000, (3, 234, 234), steps=256, target=251, transform=inv_normalize, verbose=2)
#               .on_val().to_file('dalmation.png'),
#     imaging.ClassAppearanceModel(1000, (3, 234, 234), steps=256, target=543, transform=inv_normalize, verbose=2)
#               .on_val().to_file('dumbbell.png'),
#     imaging.ClassAppearanceModel(1000, (3, 234, 234), steps=256, target=951, transform=inv_normalize, verbose=2)
#               .on_val().to_file('lemon.png'),
#     imaging.ClassAppearanceModel(1000, (3, 234, 234), steps=256, target=968, transform=inv_normalize, verbose=2)
#               .on_val().to_file('cup.png')
# ])
# trial.for_val_steps(1).to('cuda')
# trial.evaluate()
