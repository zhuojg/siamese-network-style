import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import tqdm
import datetime
from torch.optim import SGD
import torch.utils.data.dataset
import random
from PIL import Image
from torch.autograd import Variable
from torch.nn import BCELoss
import math
import numpy as np
from siamese_network import SiameseNetwork


class ClassifierLoader(torch.utils.data.dataset.Dataset):
    def __init__(self, data_path, pos_path, neg_path):
        self.data = []
        self.data_path = data_path
        with open(pos_path, 'r') as f:
            for line in f.readlines():
                data = line.split(',')
                self.data.append(data)
        with open(neg_path, 'r') as f:
            for line in f.readlines():
                data = line.split(',')
                self.data.append(data)
        random.shuffle(self.data)

    def __getitem__(self, index):
        img1, label = self.data[index][0], int(self.data[index][1])
        im1 = Image.open(os.path.join(self.data_path, img1))

        im1 = im1.convert('RGB')

        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        im1 = data_transforms(im1)

        return im1, label

    def __len__(self):
        return len(self.data)


class SiameseNetworkClassifier(nn.Module):
    def __init__(self, pre_trained_path):
        super(SiameseNetworkClassifier, self).__init__()
        self.pre_trained_siamese_network = SiameseNetwork()
        self.pre_trained_siamese_network.load_state_dict(torch.load(pre_trained_path)['model_state_dict'])
        self.target_feature = nn.Sequential(*list(self.pre_trained_siamese_network.children())[:-3])
        self.fc = nn.Linear(in_features=1024, out_features=1)
        self.drop = nn.Dropout()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        h = x
        h = self.target_feature(h)
        h = self.drop(self.fc(h))
        h = self.sig(h)
        return h


class Trainer(object):
    def __init__(self, model, optimizer, train_loader, val_loader, out_path, max_iter):
        self.model = model
        self.opt = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out_path = out_path
        self.max_iter = max_iter
        self.timestamp_start = datetime.datetime.now()

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        self.log_train_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'elapsed_time',
        ]

        self.log_val_headers = [
            'epoch',
            'iteration',
            'val/loss',
            'val/acc',
            'elapsed_time',
        ]

        if not os.path.exists(os.path.join(self.out_path, 'log_train.csv')):
            with open(os.path.join(self.out_path, 'log_train.csv'), 'w') as f:
                f.write(','.join(self.log_train_headers) + '\n')

        if not os.path.exists(os.path.join(self.out_path, 'log_val.csv')):
            with open(os.path.join(self.out_path, 'log_val.csv'), 'w') as f:
                f.write(','.join(self.log_val_headers) + '\n')

        self.epoch = 0
        self.iteration = 0

    def validate(self):
        training = self.model.training
        self.model.eval()

        val_loss = 0.
        acc_all = []

        for batch_idx, (img1, img2, is_same) in tqdm.tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                desc='Validation Epoch=%d' % self.epoch,
                ncols=80,
                leave=False
        ):
            img1, img2, is_same = Variable(img1), Variable(img2), Variable(is_same)
            is_same = torch.tensor(is_same, dtype=torch.float32)
            img1, img2, is_same = img1.cuda(), img2.cuda(), is_same.cuda()

            with torch.no_grad():
                result = self.model(img1, img2).cuda().squeeze(1)
                # result = self.model(img1, img2).squeeze(1)

            loss_fn = BCELoss(weight=None, reduce=True)
            loss = loss_fn(result, is_same)
            val_loss += loss

            acc = 0.
            is_same = is_same.cpu()
            result = result.cpu()
            for index, item in enumerate(result):
                if item.item() >= 0.5 and is_same[index].item() == 1:
                    acc += 1
                elif item.item() <= 0.5 and is_same[index].item() == 0:
                    acc += 1

            acc /= 64

            acc_all.append(acc)

        acc_all = np.array(acc_all)
        print('Val Acc=%s' % (str(acc_all.mean())))

        with open(os.path.join(self.out_path, 'log_val.csv'), 'a') as f:
            elapsed_time = (datetime.datetime.now() - self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration, val_loss, acc_all.mean(), elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.opt.state_dict(),
            'model_state_dict': self.model.state_dict(),
        }, os.path.join(self.out_path, 'checkpoint.pth.tar'))

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        epoch_loss = 0.

        acc_all = []

        for batch_idx, (img1, img2, is_same) in tqdm.tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc='Train Epoch=%d' % self.epoch,
                ncols=80,
                leave=False
        ):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue
            self.iteration = iteration
            self.opt.zero_grad()

            img1, img2, is_same = Variable(img1), Variable(img2), Variable(is_same)
            is_same = torch.tensor(is_same, dtype=torch.float32)
            img1, img2, is_same = img1.cuda(), img2.cuda(), is_same.cuda()

            result = self.model(img1, img2).cuda().squeeze(1)
            # result = self.model(img1, img2).squeeze(1)

            loss_fn = BCELoss(weight=None, reduce=True)
            loss = loss_fn(result, is_same)
            try:
                loss.backward()
                self.opt.step()
            except Exception as e:
                print(e)

            epoch_loss += loss.detach().cpu().numpy()

            if self.iteration > 0 and self.iteration % 3 == 0:
                acc = 0.
                is_same = is_same.cpu()
                result = result.cpu()
                for index, item in enumerate(result):
                    if item.item() > 0.5 and is_same[index].item() == 1:
                        acc += 1
                    elif item.item() < 0.5 and is_same[index].item() == 0:
                        acc += 1

                acc /= 64

                acc_all.append(acc)

                print('Train Acc=%s' % str(np.array(acc_all).mean()))

            if self.iteration >= self.max_iter:
                break

        with open(os.path.join(self.out_path, 'log_train.csv'), 'a') as f:
            elapsed_time = (datetime.datetime.now() - self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration, epoch_loss, np.array(acc_all).mean(), elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            self.validate()
            assert self.model.training


if __name__ == '__main__':
    train_loader = torch.utils.data.DataLoader(
        ClassifierLoader(
            data_path='../style_data_clean',
            pos_path='./modern_train.txt',
            neg_path='./non_modern_train.txt'
        ),
        batch_size=64,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        ClassifierLoader(
            data_path='../style_data_clean',
            pos_path='./modern_val.txt',
            neg_path='./non_modern_val.txt'
        ),
        batch_size=64,
        shuffle=True
    )

    # model
    model = SiameseNetworkClassifier(pre_trained_path='./data/190710_SiameseNetwork_modern.pth.tar')

    # optimizer
    opt = SGD(
        model.parameters(),
        lr=1e-4,
        momentum=0.7
    )

    # train
    trainer = Trainer(
        model=model,
        optimizer=opt,
        train_loader=train_loader,
        val_loader=val_loader,
        out_path='./log',
        max_iter=100000
    )

    trainer.epoch = 0
    trainer.iteration = 0
    trainer.train()
