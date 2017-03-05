import torch
from load_image import *
from torch.autograd import Variable
import torchvision.transforms as transforms
from resnet import *
from torchvision.utils import save_image
import torch.nn as nn
#import torch.utils.data.DataLoader
#import torchvision.datasets.coco
# How to extract features of an image from a trained model : https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/2

if __name__ == "__main__":
    data_folder = 'data/102flowers/jpg/'
    # Load image from file
    image = load_image(data_folder + 'image_00001.jpg')
    # TODO: implement a dataset class for flowers
    # Model initialization
    model = resnet18(pretrained=True)
    #alexnet = models.alexnet()
    #vgg16 = models.vgg16()

    #Perform transform
    transform = transforms.Compose([
        transforms.Scale(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])
    #transform = transforms.ToTensor()
    image = transform(image)
    save_image(image, 'test.jpg')

    image.resize_((1,)+image.size())

    image = Variable(image)
    feature = model.forward(image, until_layer='fc')
    print feature