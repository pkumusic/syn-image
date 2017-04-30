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
from models.image_embedder import ImageEncoder

from timeit import default_timer as timer

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
    trn_loader = util.get_data(opt, train_flag=True)

    alpha_num = len(util.alphabet)
    cnnDim = opt.cnnDim
    netG, netD = DCGAN(opt)
    netText = TextEncoder(alpha_num, cnnDim, opt.nz, opt.wordDim)
    print("Loading ImageEncoder...")
    start = timer()
    netImage = ImageEncoder(opt.nz)
    end = timer()
    print("ImageEncoder loading finished {:.2}m...".format((end - start) / 60.))
    start = end


    # ## test
    # batch = trn_loader.getbatch(0)
    # # print batch[0]
    # outputs = []
    # for test_data in batch[0][1]:
    #     # print batch[0][1]
    #     # print test_data.size(), test_data.size(1)
    #
    #     # expand_test_data = torch.LongTensor(1, 201)
    #     # expand_test_data[:,:test_data.size(1)] = test_data
    #     # test_data = expand_test_data
    #
    #     test_data_var = Variable(test_data)
    #     output = netText(test_data_var)
    #     outputs.append(output)
    # # output = torch.cat(outputs, dim=0)
    # output = torch.cat(outputs, 0)
    # output = torch.sum(output, dim=0)
    # print(output.size())
    # raw_input()
    # ## end test

    criterion = nn.BCELoss()
    recon_criterion = nn.MSELoss()

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    nc = int(opt.nc)
    # ngf = int(opt.ngf)
    # ndf = int(opt.ndf)


    # Set up Variable for gpu
    input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
    ref_input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
    caption = torch.FloatTensor(opt.batchSize, nz)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        netText.cuda()
        netImage.cuda()
        criterion.cuda()
        recon_criterion.cuda()
        input, label = input.cuda(), label.cuda()
        ref_input = ref_input.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
        caption = caption.cuda()

    input = Variable(input)
    label = Variable(label)
    noise = Variable(noise)
    fixed_noise = Variable(fixed_noise)
    caption = Variable(caption)
    ref_input = Variable(ref_input)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerRecon = optim.Adam(list(netG.parameters()) + list(netText.parameters()) + list(netImage.parameters())
                                , lr=opt.lr, betas=(opt.beta1, 0.999))

    len_trn = len(trn_loader)
    num_batch = len_trn // trn_loader.batch_size
    # num_batch = 60  # TEST
    # training process
    for epoch in range(opt.niter):
        for i in xrange(num_batch):
            # data = trn_loader.getbatch(i)
            data, ref_data = trn_loader.get_batch_pair(i)
            real_cpu, texts_list = trn_loader.strip_batch(data)
            ref_real_cpu, ref_texts_list = trn_loader.strip_batch(ref_data)

            real_cpu = torch.cat(real_cpu, 0)
            ref_real_cpu = torch.cat(ref_real_cpu, 0)
            batch_size = real_cpu.size(0)
            # TODO: should do padding for the sake of efficiency
            if opt.cuda:
                caption_list = [netText(Variable(text.cuda())) for text_list in texts_list for text in text_list]
                ref_caption_list = [netText(Variable(text.cuda())) for text_list in ref_texts_list for text in text_list]
            else:
                caption_list = [netText(Variable(text)) for text_list in texts_list for text in text_list]
                ref_caption_list = [netText(Variable(text)) for text_list in ref_texts_list for text in text_list]

        # for i, data in enumerate(trn_loader, 0):
            ############################
            # (0) Update G network: minimize |G(t_2-t_1+i_1) - i_2|^2
            ###########################
            optimizerRecon.zero_grad()
            # print 'real_cpu', real_cpu.size(), type(real_cpu)
            input.data.resize_(real_cpu.size()).copy_(real_cpu) # namely i_2
            ref_input.data.resize_(ref_real_cpu.size()).copy_(ref_real_cpu)
            i_1 = netImage(ref_input)
            # get text tensor
            caption_tensor = torch.cat(caption_list, 0).view(batch_size, -1, nz)
            t_2 = torch.mean(caption_tensor, 1).view(batch_size, nz, 1, 1)
            ref_caption_tensor = torch.cat(ref_caption_list, 0).view(batch_size, -1, nz)
            t_1 = torch.mean(ref_caption_tensor, 1).view(batch_size, nz, 1, 1)
            recon_i_2 = netG(t_2-t_1+i_1)
            recon_err = recon_criterion(recon_i_2.view(batch_size, -1), input.view(batch_size, -1))
            recon_err.backward()
            optimizerRecon.step()
            # print 'recon_i_2', recon_i_2.size()
            # raw_input()


            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
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

            print('[%d/%d][%d/%d] Loss_Recon: %.4f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(trn_loader),
                     recon_err.data[0], errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
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
        torch.save(netText.state_dict(), '%s/netText_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netImage.projector.state_dict(), '%s/netImage_epoch_%d.pth' % (opt.outf, epoch))

