# encoding: utf-8
#!/bin/python

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable

import util
from model_interface import DCGAN
from models.charater_embedder import TextEncoder

from tensorboard_logger import configure, log_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text2Fig Generator')
    parser = util.get_parser(parser)

    opt = parser.parse_args()
    print(opt)

    # tensorboard creation
    configure(opt.tensorboardPath)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Get data
    trn_dataset = util.get_data(opt, train_flag=True)
    trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                             batch_size=opt.batchSize,
                                             shuffle=True,
                                             num_workers=int(opt.workers))

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
    #                                          shuffle=True, num_workers=int(opt.workers))
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    alpha_num = len(alphabet)
    cnnDim = opt.cnnDim
    netG, netD = DCGAN(opt)
    netText = TextEncoder(alpha_num, cnnDim, opt.nz)

    # # test
    # test_data = torch.FloatTensor(10, 201, alpha_num)
    # # print 'test_data size', test_data.size()
    # test_data_var = Variable(test_data)
    # output = netText(test_data_var)
    # print(output.size())
    # raw_input()

    criterion = nn.BCELoss()

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    nc = int(opt.nc)
    # ngf = int(opt.ngf)
    # ndf = int(opt.ndf)


    # Set up Variable for gpu
    input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        criterion.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    input = Variable(input)
    label = Variable(label)
    noise = Variable(noise)
    fixed_noise = Variable(fixed_noise)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # training process
    for epoch in range(opt.niter):
        for i, data in enumerate(trn_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            label.data.resize_(batch_size).fill_(real_label)

            output = netD(input)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            noise.data.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
            fake = netG(noise)
            label.data.fill_(fake_label)
            output = netD(fake.detach()) # not updating netG
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.data.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(trn_loader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            log_value('Loss_D', errD.data[0], epoch * len(trn_loader) + i)
            log_value('Loss_G', errG.data[0], epoch * len(trn_loader) + i)
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                                  '%s/real_samples.png' % opt.outf,
                                  normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.data,
                                  '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                                  normalize=True)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

