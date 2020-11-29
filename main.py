"""
Mixed precision usage simple example.
"""
import argparse
import time

import numpy as np
import torch
import torchvision.transforms as transforms

from apex import amp
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from networks import Classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, choices=['resnet34', 'resnet50'], default='resnet34',
                        help='Architecture to train.')
    parser.add_argument('--batch-size', type=int, required=True, help='Batch size to train NN.')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train NN.')
    parser.add_argument('--use-mixed-precision', type=str, default="O0", choices=["O0", "O1"],
                        help='Disable or enable mixed precision training.')

    args = parser.parse_args()

    transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])

    train_dataset = CIFAR100(root='./', download=True, train=True, transform=transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda:0")
    net = Classifier(backbone=args.arch, num_classes=100).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

    net, optimizer = amp.initialize(net, optimizer, opt_level=args.use_mixed_precision)

    epoch_train_loss = []
    for epoch in range(args.epochs):
        start = time.time()
        for images, labels in train_dataloader:
            optimizer.zero_grad()

            images, labels = images.to(device), labels.to(device)
            predictions = net(images)

            loss = criterion(predictions, labels)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()
            epoch_train_loss.append(loss.item())
        finish = time.time()
        print(f"Epoch: {epoch}, train loss: {np.mean(epoch_train_loss):.5f}, epoch time: {finish - start}")
