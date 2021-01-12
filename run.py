import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data

from model import *
import argparse
from args import argument_parser

parser = argument_parser()
args = parser.parse_args()

if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor()])
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    if args.dataset=='mnist':
        train_dataset = datasets.MNIST(root=args.data_path, train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, transform=transform, download=True)
    elif args.dataset =='f-mnist':
        train_dataset = datasets.FashionMNIST(root=args.data_path, train=True, transform=transform, download=True)
        test_dataset = datasets.FashionMNIST(root=args.data_path, train=False, transform=transform, download=True)
    else:
        print('The dataset \'%s\' is not suppoerted!' %(args.dataset))

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # model
    if args.model=='AE':
        model = AutoEncoder()
    elif args.model =='DCW':
        model = DCW()
    elif args.model =='DCW_woConstraint':
        model = DCW_woConstraint()
    else:
        print("The model \'%s\' is not supported!"% (args.model))
    
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.MSELoss()

    # training
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 784)
            out = model(images)

            optim.zero_grad()
            loss = criterion(out, images)
            loss.backward()
            optim.step()
        print('[{}/{}] Loss:'.format(epoch+1, epochs), loss.item())

    # testing
    images, labels = next(iter(test_loader))
    images = images.view(-1, 784)
    model.eval()
    out = model(images).view(batch_size, 1, 28, 28)
    visulization(out.detach(), labels, args.result_path)