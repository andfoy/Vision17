#! /usr/bin/env python

from __future__ import print_function

import torch
import argparse
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from textures import TextureLoader

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Texture'
                                             ' Classification Net')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before'
                    'logging training status')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')

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


class Net(nn.Module):
    def __init__(self, num_classes=25):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512 * 7 * 7, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def embed(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))

        # print("CONV5 {0}".format(x.size()))
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        # x = F.dropout()
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, x):
        # print("In: {0}".format(x.size()))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))

        # print("CONV5 {0}".format(x.size()))
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        # x = F.dropout()
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print("CONV1: {0}".format(x.size()))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        # print("Reshape: {0}".format(x.size()))
        # x = F.relu(self.fc1(x))
        # print("FC1: {0}".format(x.size()))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.fc2(x))
        # print("FC2: {0}".format(x.size()))
        return F.log_softmax(x)


load_ext = False
model = Net()
if osp.exists(args.save):
    with open(args.save, 'rb') as f:
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

# optimizer = optim.Adam(model.parameters(), lr=args.lr)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch, lr=args.lr):
    model.train()

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for batch_idx, (data, target, _) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target, _ in val_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    # loss function already averages over batch size
    test_loss /= len(test_loader.dataset)
    print('\nVal set: Average loss: {:.4f},'
          'Accuracy: {}/{} ({:.0f}%)\n'.format(
              test_loss, correct, len(test_loader.dataset),
              100. * correct / len(test_loader.dataset)))
    return test_loss


def write_predictions():
    model.eval()
    indices = np.zeros(len(test_loader.dataset))
    predictions = np.zeros(len(test_loader.dataset))
    head = 0
    offset = args.batch_size
    for batch, (data, _, idx)in enumerate(test_loader):
        print("Test batch: {0}".format(batch))
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        output = model(data)
        pred = output.data.max(1)[1] + 1
        indices[head:head + offset] = idx.cpu().numpy().ravel()
        predictions[head:head + offset] = pred.cpu().numpy().ravel()
        head += offset
    submission = np.vstack((indices, predictions)).T
    np.savetxt('test.csv', submission, delimiter=",", fmt="%d")


if __name__ == '__main__':
    lr = args.lr
    if not load_ext:
        best_val_loss = None
    else:
        best_val_loss = test(0)
    try:
        for epoch in range(1, args.epochs + 1):
            train(epoch, lr)
            val_loss = test(epoch)
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_val_loss = val_loss
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        # model = Net()
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)

    test(epoch)
