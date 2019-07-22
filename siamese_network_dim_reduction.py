import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import tqdm
import datetime
from torch.optim import SGD
import torch.utils.data.dataset
from PIL import Image
from torch.autograd import Variable
from torch.nn import BCELoss
import math
import numpy as np
from torchsummary import summary


class SiameseNetworkDR(nn.Module):
    def __init__(self, out_dim):
        super(SiameseNetworkDR, self).__init__()
        self.pretrained_model = models.resnet34(pretrained=True)
        self.cnn = nn.Sequential(*list(self.pretrained_model.children())[:-2])
        self.fc = nn.Linear(in_features=512 * 7 * 7, out_features=1024)
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features=2048, out_features=out_dim)

    def forward(self, x):
        output = self.cnn(x)
        output = output.view((-1, 512 * 7 * 7))
        output = self.drop(self.fc(output))

        return output


class SiameseLoader(torch.utils.data.dataset.Dataset):
    def __init__(self, data_path, txt_path):
        self.data = []
        self.data_path = data_path
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                data = line.split(',')
                self.data.append(data)

    def __getitem__(self, index):
        img1, img2, is_same = self.data[index][0], self.data[index][1], int(self.data[index][2])
        im1 = Image.open(os.path.join(self.data_path, img1))
        im2 = Image.open(os.path.join(self.data_path, img2))

        im1 = im1.convert('RGB')
        im2 = im2.convert('RGB')

        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        im1 = data_transforms(im1)
        im2 = data_transforms(im2)

        return im1, im2, is_same

    def __len__(self):
        return len(self.data)


def contrastive_loss(self, x1, x2, y):
    margin = 1.
    d = torch.pairwise_distance(x1, x2, keepdim=False)
    d = d.tolist()
    y = y.tolist()
    loss = []

    for i, item in enumerate(d):
        loss.append(0.5 * y[i] * item * item + 0.5 * (1-y[i]) * max(0, (margin - item)) * max(0, (margin - item)))

    loss = np.array(loss)

    loss_mean = torch.Tensor([loss.mean()])

    return Variable(loss_mean, requires_grad=True)


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        return

    def forward(self, x1, x2, y):
        d = torch.pairwise_distance(x1, x2, keepdim=False)

        margin = 1.
        d = torch.pairwise_distance(x1, x2, keepdim=False)

        zero_tensor = torch.Tensor([0]*d.shape[0])

        loss = 0.5 * y * d * d + 0.5 * (1-y) * torch.max(zero_tensor, margin - d) * torch.max(zero_tensor, margin - d)

        return torch.mean(loss)


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
            # 'train/acc',
            'elapsed_time',
        ]

        self.log_val_headers = [
            'epoch',
            'iteration',
            'val/loss',
            # 'val/acc',
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

        # initialization visdom
        # if not use, just comment them
        # self.viz = visdom.Visdom()
        # self.viz.text('train_loss', win='train_loss')
        # self.viz.text('val_loss', win='val_loss')

    def validate(self):
        training = self.model.training
        self.model.eval()

        val_loss = 0.
        # acc_all = []

        for batch_idx, (img1, img2, is_same) in tqdm.tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                desc='Validation Epoch=%d' % self.epoch,
                ncols=80,
                leave=False
        ):
            img1, img2, is_same = Variable(img1), Variable(img2), Variable(is_same)
            is_same = torch.tensor(is_same, dtype=torch.float32)
            # img1, img2, is_same = img1.cuda(), img2.cuda(), is_same.cuda()

            with torch.no_grad():
                # result1 = self.model(img1).cuda().squeeze(1)
                # result2 = self.model(img2).cuda().squeeze(1)
                result1 = self.model(img1).squeeze(1)
                result2 = self.model(img2).squeeze(1)

            loss_fn = ContrastiveLoss()
            loss = loss_fn(result1, result2, is_same)
            val_loss += loss

            # acc = 0.
            # is_same = is_same.cpu()
            # result = result.cpu()
            # for index, item in enumerate(result):
            #     if item.item() >= 0.5 and is_same[index].item() == 1:
            #         acc += 1
            #     elif item.item() <= 0.5 and is_same[index].item() == 0:
            #         acc += 1
            #
            # acc /= 64
            #
            # acc_all.append(acc)

        # acc_all = np.array(acc_all)
        # print('Val Acc=%s' % (str(acc_all.mean())))

        with open(os.path.join(self.out_path, 'log_val.csv'), 'a') as f:
            elapsed_time = (datetime.datetime.now() - self.timestamp_start).total_seconds()
            # log = [self.epoch, self.iteration, val_loss, acc_all.mean(), elapsed_time]
            log = [self.epoch, self.iteration, val_loss, elapsed_time]
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

        # acc_all = []

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
            # img1, img2, is_same = img1.cuda(), img2.cuda(), is_same.cuda()

            # result1 = self.model(img1).cuda().squeeze(1)
            # result2 = self.model(img2).cuda().squeeze(1)
            # result = self.model(img1, img2).squeeze(1)
            result1 = self.model(img1).squeeze(1)
            result2 = self.model(img2).squeeze(1)

            loss_fn = ContrastiveLoss()
            loss = loss_fn(result1, result2, is_same)

            try:
                loss.backward()
                self.opt.step()
            except Exception as e:
                print(e)

            epoch_loss += loss.detach().cpu().numpy()
            #
            # if self.iteration > 0 and self.iteration % 3 == 0:
            #     acc = 0.
            #     is_same = is_same.cpu()
            #     result = result.cpu()
            #     for index, item in enumerate(result):
            #         if item.item() > 0.5 and is_same[index].item() == 1:
            #             acc += 1
            #         elif item.item() < 0.5 and is_same[index].item() == 0:
            #             acc += 1
            #
            #     acc /= 64
            #
            #     acc_all.append(acc)
            #
            #     print('Train Acc=%s' % str(np.array(acc_all).mean()))

            if self.iteration >= self.max_iter:
                break

        with open(os.path.join(self.out_path, 'log_train.csv'), 'a') as f:
            elapsed_time = (datetime.datetime.now() - self.timestamp_start).total_seconds()
            # log = [self.epoch, self.iteration, epoch_loss, np.array(acc_all).mean(), elapsed_time]
            log = [self.epoch, self.iteration, epoch_loss, elapsed_time]
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
        SiameseLoader(
            data_path='../style_data_clean',
            txt_path='./modern_pairs_train.txt'
        ),
        batch_size=8,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        SiameseLoader(
            data_path='../style_data_clean',
            txt_path='./modern_pairs_val.txt'
        ),
        batch_size=8,
        shuffle=True
    )

    # model
    model = SiameseNetworkDR(32)

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

