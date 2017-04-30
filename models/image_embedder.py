#!/usr/bin/env python
# encoding: utf-8

import torchvision.models as models
import torch.nn as nn

# ImageEncoder
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(ImageEncoder, self).__init__()
        self.encoder = models.vgg16(pretrained=True).features
        # self.projection = nn.Linear(300, embed_dim)
        self.projector = nn.Sequential( # 64 input picture size
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, embed_dim),
        )
        # self.projector = nn.Sequential( # 224 input picture size
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, embed_dim),
        # )
        self._initialize_weights()

    def _initialize_weights(self):
        # do not update the encoder part
        for m in self.projector.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        # print 'feature size', x.size()
        x = self.projector(x)
        return x
