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
from model_interface import DCGAN, WGAN
#  from models.charater_embedder import TextEncoder

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
    netG, netD = WGAN(opt)

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
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    input = Variable(input)
    label = Variable(label)
    noise = Variable(noise)
    fixed_noise = Variable(fixed_noise)

    # setup optimizer
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr)

    # training process
    for epoch in range(opt.niter):
        for i, data in enumerate(trn_loader, 0):
            ############################
            # (1) Update D network: maximize E(D(x)) + E(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            label.data.resize_(batch_size).fill_(real_label)

            output = netD(input)
            D_x = torch.mean(output)
            # train with fake
            noise.data.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)
            fake = netG(noise)
            output = netD(fake.detach()) # not updating netG
            D_G_z1 = torch.mean(output)
            errD = -(D_x - D_G_z1)
            errD.backward()
            optimizerD.step()
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

            ############################
            # (2) Update G network: maximize E(D(G(z)))
            ###########################
            netG.zero_grad()
            output = netD(fake)
            errG = -torch.mean(output)
            errG.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f'
                  % (epoch, opt.niter, i, len(trn_loader),
                     errD.data.cpu().numpy(), errG.data.cpu().numpy(), D_x.data.cpu().numpy(), D_G_z1.data.cpu().numpy()))
            #log_value('Loss_D', errD.data[0], epoch * len(trn_loader) + i)
            #log_value('Loss_G', errG.data[0], epoch * len(trn_loader) + i)
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

