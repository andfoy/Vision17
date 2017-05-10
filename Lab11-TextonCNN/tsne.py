#! /usr/bin/env python

from __future__ import print_function

import argparse
import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torchvision import models

from textures import TextureLoader

from main import Net
from sklearn import manifold
from scipy.spatial.distance import pdist, squareform


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Texture'
                                             ' Classification Net')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
# parser.add_argument('--n-iter', type=int, default=500, metavar='N',
#                     help='number of T-SNE iterations (default: 500)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--perplexity', type=float, default=30, metavar='M',
                    help='T-SNE perplexity (default: 30)')
parser.add_argument('--n-dim', type=int, default=2, metavar='M',
                    help='T-SNE number of dimensions (default: 2)')
parser.add_argument('--n-topics', type=int, default=2, metavar='M',
                    help='T-SNE number of dimensions (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before'
                    'logging training status')
parser.add_argument('--model', type=str, default='model.pt',
                    help='path to file that contains model parameters')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(
    TextureLoader('data', train=True, download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.485,), (0.229,))
                  ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


test_loader = torch.utils.data.DataLoader(
    TextureLoader('data', test=True, download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.485,), (0.229,))
                  ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


val_loader = torch.utils.data.DataLoader(
    TextureLoader('data', download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.485,), (0.229,))
                  ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


def cnn_codes_to_distance(x, model):
    codes = model.embed(x)
    codes = codes.data
    if args.cuda:
        codes = codes.cpu()
    distances = pdist(codes.numpy())
    points = manifold.t_sne._joint_probabilities(distances,
                                                 args.perplexity,
                                                 False)
    points = squareform(points)
    return points


def pairwise(data):
    n_obs, dim = data.size()
    xk = data.unsqueeze(0).expand(n_obs, n_obs, dim)
    xl = data.unsqueeze(1).expand(n_obs, n_obs, dim)
    dkl2 = ((xk - xl)**2.0).sum(2).squeeze()
    return dkl2


class VTSNE(nn.Module):
    def __init__(self, n_points, n_topics, n_dim):
        self.n_points = n_points
        self.n_dim = n_dim
        super(VTSNE, self).__init__()
        # Logit of datapoint-to-topic weight
        self.logits_mu = nn.Embedding(n_points, n_topics)
        self.logits_lv = nn.Embedding(n_points, n_topics)

    @property
    def logits(self):
        return self.logits_mu

    def reparametrize(self, mu, logvar):
        # From VAE example
        # https://github.com/pytorch/examples/blob/master/vae/main.py
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)
        kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kld = torch.sum(kld).mul_(-0.5)
        return z, kld

    def sample_logits(self, i=None):
        if i is None:
            return self.reparametrize(self.logits_mu.weight,
                                      self.logits_lv.weight)
        else:
            return self.reparametrize(self.logits_mu(i), self.logits_lv(i))

    def forward(self, pij, i, j):
        # Get  for all points
        x, loss_kldrp = self.sample_logits()
        # Compute squared pairwise distances
        dkl2 = pairwise(x)
        # Compute partition function
        n_diagonal = dkl2.size()[0]
        part = (1 + dkl2).pow(-1.0).sum() - n_diagonal
        # Compute the numerator
        xi, _ = self.sample_logits(i)
        xj, _ = self.sample_logits(j)
        num = ((1. + (xi - xj)**2.0).sum(1)).pow(-1.0).squeeze()
        qij = num / part.expand_as(num)
        # Compute KLD(pij || qij)
        loss_kld = pij * (torch.log(pij) - torch.log(qij))
        # Compute sum of all variational terms
        return loss_kld.sum() + loss_kldrp.sum() * 1e-7

    def __call__(self, *args):
        return self.forward(*args)


load_ext = False
model = Net()
tsne = VTSNE(len(train_loader.dataset), args.n_topics, args.n_dim)
if osp.exists(args.model):
    with open(args.model, 'rb') as f:
        state_dict = torch.load(f)
        discard = [x for x in state_dict if x.startswith('fc1')]
        state = model.state_dict()
        state.update(state_dict)
        try:
            model.load_state_dict(state)
        except Exception:
            for key in discard:
                state_dict.pop(key)
            state = model.state_dict()
            state.update(state_dict)
            model.load_state_dict(state)
    load_ext = True

if args.cuda:
    model.cuda()
    tsne.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train_tsne(epoch):
    tsne.train()
    total = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        points_2d = cnn_codes_to_distance(data, model)
        points = points_2d.ravel().astype('float32')

        pos_y, pos_x = np.indices(points_2d.shape)
        pos_y = pos_y.ravel()
        pos_x = pos_x.ravel()

        idx = pos_y != pos_x
        points = Variable(torch.FloatTensor(points[idx]))
        pos_y = Variable(torch.FloatTensor(pos_y[idx]))
        pos_x = Variable(torch.FloatTensor(pos_x[idx]))

        if args.cuda:
            points = points.cuda()
            pos_x = pos_x.cuda()
            pos_y = pos_y.cuda()

        optimizer.zero_grad()
        loss = tsne(points, pos_y, pos_x)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train_tsne(epoch)
