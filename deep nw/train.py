from __future__ import print_function

import os
import random
import math
import torch
import torch.nn as nn
# import torch.legacy.nn as lnn
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import pickle

import torch.nn.functional as F
from torch.distributions.normal import Normal


def _train(opt, net, criterion, dataloader, testloader, N=2400):
    # device = torch.device("cuda" if opt.cuda else "cpu")
    try:
        try:
            os.makedirs(opt.outf)
        except OSError:
            pass
        os.makedirs(opt.outf + '/saved_models')
    except OSError:
        pass

    if opt.cuda:
        net.cuda()
        criterion.cuda()

    # setup optimizer
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    print("Starting training...\n")

    for epoch in range(opt.niter):
        err_batch = []
        net.train()
        for i, (input, label) in enumerate(dataloader, 0):
            net.zero_grad()
            # print(label)
            # print(input)
            # label = one_hot_embedding(label, opt.numClasses)  # one-hot encoding
            # print("one-hot: ", label)
            input = Variable(input)
            label = Variable(label)
            if opt.cuda:
                input, label = input.cuda(), label.cuda()
            # batch_size = real_cpu.size(0)
            label = label.long()

            output = net(input)
            # print(output.shape, output, label)
            # print(output)
            output = output.squeeze()
            loss = criterion(output, label)

            loss.backward(retain_graph=True)
            optimizer.step()
            ##printing statistics:
            if (i + 1) % np.floor(N / opt.batchSize) == 0:
                loss_data = loss.data[0]
                err_batch.append(loss_data)
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                      % (epoch + 1, opt.niter, i + 1, N // opt.batchSize, loss_data))

                if (epoch + 1) % opt.saveInt == 0 and epoch != 0:
                    torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.outf+'/saved_models', epoch+1))

                ##printing train statistics:
                # Test the Model
                correct = 0
                total = 0
                for images, labels in dataloader:
                    if opt.cuda:
                        images, labels = images.cuda(), labels.cuda()
                    images = Variable(images)  # .view(-1, 28*28))
                    outputs = net(images)
                    # print(outputs.shape, output)
                    # _, predicted = torch.max(outputs.data, 1)
                    predicted = torch.argmax(outputs, dim=1)
                    # print("predicted :", predicted.cpu().detach().numpy().shape[0])
                    if epoch % 2 == 0:
                        pass
                        # print("labels: ", labels, "outputs: ", outputs, "predicted: ", predicted)
                    # labels = torch.max(labels.float(), 1)[1]
                    ##    predicted = torch.round(outputs.data).view(-1).long()
                    total += labels.size(0)
                    correct += (predicted.float() == labels.float()).sum()

                # print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))

                # ##printing test statistics:
                # # Test the Model
                correct = 0
                total = 0
                for images, labels in testloader:
                    if opt.cuda:
                        images, labels = images.cuda(), labels.cuda()
                    images = Variable(images)  # .view(-1, 28*28))
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)

                    # labels = torch.max(labels.float(), 1)[1]
                    ##    predicted = torch.round(outputs.data).view(-1).long()
                    total += labels.size(0)
                    correct += (predicted.float() == labels.float()).sum()

                print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

