#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

# CNN/RNN TextEncoder
class TextEncoder(nn.Module):
    def __init__(self, alphasize, cnn_dim, embed_dim, word_dim):
        super(TextEncoder, self).__init__()
        self.cnn = TextEncoderCNN(alphasize, cnn_dim, word_dim)
        self.rnn = TextEncoderRNN(cnn_dim, embed_dim)

    def forward(self, input):
        """ input: batch x seq_len x input_size"""
        input = self.cnn.forward(input)
        # output: batch x cnn_dim x reduced_seq_len
        input = torch.transpose(input, 1, 2)
        input = torch.transpose(input, 0, 1)
        # output: reduced_seq_len x batch x cnn_dim
        output = self.rnn.forward(input)
        return output

class TextEncoderCNN(nn.Module):
    def __init__(self, alphasize, cnn_dim, word_dim):
        super(TextEncoderCNN, self).__init__()
        self.emb = nn.Embedding(alphasize, word_dim)
        # -- alphasize x 201 (201 is the imagined dimension of the sentence)
        self.main = nn.Sequential(
            # nn.Conv1d(alphasize, 384, 4),
            nn.Conv1d(word_dim, 384, 4),
            nn.Threshold(1e-6, 0),
            nn.MaxPool1d(3, 3),
            # -- 384 x 66
            nn.Conv1d(384, 512, 4),
            nn.Threshold(1e-6, 0),
            nn.MaxPool1d(3, 3),
            # -- 512 x 21
            nn.Conv1d(512, cnn_dim, 4),
            nn.Threshold(1e-6, 0))
            # nn.MaxPool1d(3, 2))
            # -- cnn_dim x 8
    def forward(self, input):
        # print 'prev input', input.size()
        # TODO: since no temporal available, use the transpose and Conv1D operation
        input = self.emb.forward(input)
        input = torch.transpose(input, 1, 2)
        # output: batch x cnn_dim x reduced_seq_len
        return self.main.forward(input)


class TextEncoderRNN(nn.Module):
    def __init__(self, cnn_dim, embed_dim):
        super(TextEncoderRNN, self).__init__()
        self.main = nn.GRU(cnn_dim, embed_dim, 1)
        # input_size, embed_size, num_layer

    def forward(self, input):
        """ input: seq_len x batch x input_size"""
        output, h_n = self.main(input)
        # h_n: num_layer * num_directions x batch x embed_size
        trans = torch.transpose(h_n, 0, 1)
        return trans.view((h_n.size(1), -1))
        # final: batch x num_layer * num_directions * embed_size

