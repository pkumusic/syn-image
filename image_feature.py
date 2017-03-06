import torch
from load_image import *
from torch.autograd import Variable
import torchvision.transforms as transforms
from resnet import *
from torchvision.utils import save_image
import torch.nn as nn
import numpy as np
import argparse
#import torch.utils.data.DataLoader
#import torchvision.datasets.coco
# How to extract features of an image from a trained model : https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/2
# TODO: implement a dataset class for flowers
import glob
def flowers(args):
    data_folder = 'data/102flowers/'
    labels = load_labels(data_folder + 'imagelabels.txt')
    #  labels = load_labels(data_folder + 'imagelabels_10class.txt')
    n = len(labels)
    print 'total number: ', n
    # Model initialization
    features = []
    conv3s = []
    conv4s = []
    model = resnet152(pretrained=True)
    # alexnet = models.alexnet()
    # vgg16 = models.vgg16()
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            model = model.cuda()
    for i in xrange(n):
        #if i % 100 == 0:
        print i
        # Load image from file
        image = load_image(data_folder + 'jpg/image_%.5d.jpg'%(i+1))
        label = labels[i]

        # Perform transform
        image = image.resize((224,224))
        transform = transforms.Compose([
            # transforms.Scale((224)),
            transforms.ToTensor(),
        ])
        # transform = transforms.ToTensor()
        image = transform(image)
        #  save_image(image, 'test.jpg')

        image.resize_((1,) + image.size())

        if args.cuda:
            image = image.cuda()
        image = Variable(image)
        if args.cuda:
            feature = model.forward(image, until_layer='fc').data.cpu().numpy()
            conv3 = model.forward(image, until_layer='3').data.cpu.numpy()
            conv3 = conv3.reshape(1,-1)
            conv4 = model.forward(image, until_layer='4').data.cpu.numpy()
            conv4 = conv4.reshape(1,-1)
        else:
            feature = model.forward(image, until_layer='fc').data.numpy()
            conv3 = model.forward(image, until_layer='3').data.numpy()
            conv3 = conv3.reshape(1,-1)
            conv4 = model.forward(image, until_layer='4').data.numpy()
            conv4 = conv4.reshape(1,-1)
        #print feature
        #print feature.shape
        #exit()
        features.append(feature)
        conv3s.append(conv3)
        conv4s.append(conv4)
    features = np.concatenate(features, axis=0)
    np.save('fc_features_res152', features)
    np.save('fc_conv3_res152', conv3s)
    np.save('fc_conv4_res152', conv4s)

# def birds(args):
#     data_folder = 'data/CUB_200_2011/CUB_200_2011'
#     folders = glob.glob(data_folder + '/images/*')
#     label = 0
#     image_label_pairs = []
#     for folder in folders:
#         label += 1
#         image_paths = glob.glob(folder + '/*')
#         for image_path in image_paths:
#             image = load_image(image_path)
#             image_label_pairs.append((label, image))
#     print image_label_pairs
#
#     labels = load_labels(data_folder + 'imagelabels.txt')
#     #  labels = load_labels(data_folder + 'imagelabels_10class.txt')
#     n = len(labels)
#     print 'total number: ', n
#     # Model initialization
#     features = []
#     model = resnet152(pretrained=True)
#     # alexnet = models.alexnet()
#     # vgg16 = models.vgg16()
#     if torch.cuda.is_available():
#         if not args.cuda:
#             print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#         else:
#             model = model.cuda()
#     for i in xrange(n):
#         #if i % 100 == 0:
#         print i
#         # Load image from file
#         image = load_image(data_folder + 'jpg/image_%.5d.jpg'%(i+1))
#         label = labels[i]
#
#         # Perform transform
#         image = image.resize((224,224))
#         transform = transforms.Compose([
#             # transforms.Scale((224)),
#             transforms.ToTensor(),
#         ])
#         # transform = transforms.ToTensor()
#         image = transform(image)
#         #  save_image(image, 'test.jpg')
#
#         image.resize_((1,) + image.size())
#
#         if args.cuda:
#             image = image.cuda()
#         image = Variable(image)
#         if args.cuda:
#             feature = model.forward(image, until_layer='fc').data.cpu().numpy()
#         else:
#             feature = model.forward(image, until_layer='fc').data.numpy()
#         #print feature
#         #print feature.shape
#         #exit()
#         features.append(feature)
#     features = np.concatenate(features, axis=0)
#     np.save('fc_features_res152', features)

def extract(args):

    data_folder = 'data/102flowers/'
    labels = load_labels(data_folder + 'imagelabels_10class.txt')
    n = len(labels)
    print 'total number: ', n
    # Model initialization
    images = []
    model = resnet152(pretrained=True)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            model = model.cuda()
    for i in xrange(n):
        if i % 100 == 0:
            print i
        # Load image from file
        image = load_image(data_folder + 'jpg/image_%.5d.jpg'%(i+1))
        # alexnet = models.alexnet()
        # vgg16 = models.vgg16()
        # Perform transform
        image = image.resize((224,224))
        transform = transforms.Compose([
            # transforms.Scale((224)),
            transforms.ToTensor(),
        ])
        # transform = transforms.ToTensor()
        image = transform(image)
        image.resize_((1,) + image.size())
        images.append(image)
        #save_image(image, 'test.jpg')
    images = torch.cat(images)
    if args.cuda:
        images = images.cuda()
    images = Variable(images)

    features = model.forward(images, until_layer='fc').data.numpy()
    #print feature
    print features.shape
    #features = np.concatenate(features, axis=0)
    np.save('fc_features_res152', features)

def visualize_filter():
    model = resnet18(pretrained=True)
    for param in model.parameters():
        print (type(param.data)), param.size()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Feature Extractor')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    args = parser.parse_args()
    #  extract(args)
    flowers(args)
    #birds(args)
    #visualize_filter()


