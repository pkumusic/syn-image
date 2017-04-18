#!/usr/bin/env python
# encoding: utf-8

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

from PIL import Image
import csv
import sys
csv.field_size_limit(sys.maxsize)
from collections import defaultdict

# CONSTANTS
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

def get_parser(parser):
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--wordDim', type=int, default=100, help='dimension of the word embedding')
    parser.add_argument('--cnnDim', type=int, default=256, help='dimension of the CNN text encoder')
    parser.add_argument('--nc', type=int, default=3, help='number of channel for the input image')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='./checkpoints', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, default=123, help='manual seed')
    parser.add_argument('--tensorboardPath', type=str, default='./runs', help='path for the tensorboard logger')
    return parser

def get_data(args, train_flag=True):
    transform = transforms.Compose([transforms.Scale(args.imageSize),
                                    transforms.CenterCrop(args.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.dataset in ['imagenet', 'folder', 'lfw']:
        dataset = dset.ImageFolder(root=args.dataroot,
                                   transform=transform)

    elif args.dataset == 'lsun':
        transform = transforms.Compose([transforms.Scale(args.imageSize),
                                    transforms.CenterCrop(args.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = dset.LSUN(db_path=args.dataroot,
                            classes=['church_outdoor_train'],
                            transform=transform)

    elif args.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=args.dataroot,
                               download=True,
                               train=train_flag,
                               transform=transform)

    elif args.dataset == 'cifar100':
        dataset = dset.CIFAR100(root=args.dataroot,
                                download=True,
                                train=train_flag,
                                transform=transform)

    elif args.dataset == 'mnist':
        transform = transforms.Compose([transforms.Scale(28),
                                    transforms.CenterCrop(28),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = dset.MNIST(root=args.dataroot,
                             download=True,
                             train=train_flag,
                             transform=transform)

    elif args.dataset == 'flowers':
        # args.imageSize should be 224
        if args.imageSize != 224:
            print('Alert: imageSize not equal 224')
        transform = transforms.Compose([
            transforms.Scale(args.imageSize),
            transforms.CenterCrop(args.imageSize),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.0, 0.0, 0.0],
            #                      std=[1.0, 1.0, 1.0])])  # TODO: should normalize to range [-1, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])])  # TODO: should normalize to range [-1, 1]
        string_transform = TxtTransform(alphabet=alphabet).transform
        dataset = ImgTxt(data='/usr1/home/yuexinw/courses/11777/PData/flowers.csv', # text file TODO: changed to a variable
                         path=args.dataroot,
                         batch_size=args.batchSize,
                         transform=transform,
                         target_transform=string_transform)
    else:
        print 'Dataset not recognizable'
        print args.help
        sys.exit()
    return dataset

class TxtTransform():
    def __init__(self, alphabet):
        # a str
        self.alphabet = alphabet
        # construct a dictionary
        self.alpha2idx = {}
        for i, sym in enumerate(alphabet):
            self.alpha2idx[sym] = i
    def transform(self, list_of_str):
        ret_list = []
        for ele_str in list_of_str:
            l = len(ele_str)
            tensor = torch.LongTensor(1, l)
            for i, sym in enumerate(ele_str):
                tensor[0, i] = self.alpha2idx.get(sym)
            ret_list.append(tensor)
        return ret_list



class ImgTxt():
    def __init__(self, data, path, batch_size=3, transform=None, target_transform=None):
        #         self.birdpath = "/usr1/home/yuexinw/courses/11777/syn-image/data/CUB_200_2011/CUB_200_2011/images/"
        #         self.flowerpath = "/usr1/home/yuexinw/courses/11777/syn-image/data/102flowers/jpg/"
        self.path = path
        columns = defaultdict(list)
        f = open(data, 'r')
        reader = csv.DictReader(f)
        for row in reader:
            for (k, v) in row.iteritems():
                columns[k].append(v)
        self.ids = columns['id']
        self.text = columns['text']
        self.category = columns['category']
        self.batch_size = batch_size
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = self.text[index].split('\n')
        if 'flower' in self.path:
            path = self.path + img_id + '.jpg'
        else:
            path = self.path + self.category[index] + '/' + img_id + '.jpg'
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def getbatch(self, index):
        img_ids = self.ids[index * self.batch_size:(index + 1) * self.batch_size]
        targets = [x.split('\n') for x in self.text[index * self.batch_size:(index + 1) * self.batch_size]]
        ret = []
        for i in xrange(len(img_ids)):
            img_id = img_ids[i]
            if 'flower' in self.path:
                path = self.path + img_id + '.jpg'
            else:
                path = self.path + self.category[index] + '/' + img_id + '.jpg'
            img = Image.open(path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(targets[i])

            # ret.append((img, targets[i]))
            ret.append((img, target))

        return ret

    def strip_batch(self, batch):
        img_list = []
        target_list = []
        for img, target in batch:
            img_list.append(img.unsqueeze(0))
            target_list.append(target)
        return img_list, target_list


    def get_batch_pair(self, index):
        max_index = self.__len__() / self.batch_size # could not be achieved
        # i2 could change to other things with less correlation
        i1, i2 = index, index+1
        while True:
            if i1 >= max_index:
                i1 -= max_index
            if i2 >= max_index:
                i2 -= max_index
            p1, p2 = self.getbatch(i1), self.getbatch(i2)
            if len(p1) == len(p2):
                break
        return p1, p2
        
