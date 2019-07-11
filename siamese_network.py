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
from torchsummary import summary


class SiameseNetworkSimple(nn.Module):
    def __init__(self):
        super(SiameseNetworkSimple, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 32, 3, stride=1)
        self.sigmoid1_1 = nn.Sigmoid()

        self.conv1_2 = nn.Conv2d(32, 64, 3, stride=1)
        self.sigmoid1_2 = nn.Sigmoid()
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv2
        self.conv2_1 = nn.Conv2d(64, 64, 3, stride=1)
        self.sigmoid2_1 = nn.Sigmoid()

        self.conv2_2 = nn.Conv2d(64, 128, 3, stride=1)
        self.sigmoid2_2 = nn.Sigmoid()
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)


        # conv3
        self.conv3_1 = nn.Conv2d(128, 128, 3, stride=1)
        self.sigmoid3_1 = nn.Sigmoid()
        self.pool3_1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3_2 = nn.Conv2d(128, 256, 3, stride=1)
        self.sigmoid3_2 = nn.Sigmoid()
        self.pool3_2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)


        # fc
        self.fc = nn.Linear(in_features=256 * 12 *12, out_features=1)
        self.sigmoid = nn.Sigmoid()

        # join
        self.join = nn.Linear(in_features=2, out_features=1)
        self.sigmoid_1 = nn.Sigmoid()

    def forward_one(self, x):
        h = x

        h = self.sigmoid1_1(self.conv1_1(h))
        h = self.sigmoid1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.sigmoid2_1(self.conv2_1(h))
        h = self.sigmoid2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.sigmoid3_1(self.conv3_1(h))
        h = self.pool3_1(h)
        h = self.sigmoid3_2(self.conv3_2(h))
        h = self.pool3_2(h)

        h = h.view((-1, 256 * 12 * 12))

        h = self.sigmoid(self.fc(h))

        return h

    def forward(self, x1, x2):
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)

        result = self.sigmoid_1(self.join(torch.cat((output1, output2), dim=1)))

        return result


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.pretrained_model = models.resnet34(pretrained=True)
        self.cnn = nn.Sequential(*list(self.pretrained_model.children())[:-2])
        self.fc = nn.Linear(in_features=512 * 7 * 7, out_features=1024)
        self.drop = nn.Dropout()
        self.acti = nn.Sigmoid()
        self.final = nn.Linear(in_features=2048, out_features=1)
        self.final_acti = nn.Sigmoid()

    def forward(self, x1, x2):
        output1 = self.cnn(x1)
        output2 = self.cnn(x2)
        output1 = output1.view((-1, 512 * 7 * 7))
        output2 = output2.view((-1, 512 * 7 * 7))
        output1 = self.acti(self.drop(self.fc(output1)))
        output2 = self.acti(self.drop(self.fc(output2)))
        result = self.final(torch.cat((output1, output2), dim=1))
        result = self.final_acti(result)

        return result


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


class SiameseRandomLoader(torch.utils.data.dataset.Dataset):
    def __init__(self, data_path, target_tag):
        self.data_path = data_path
        self.target_tag = target_tag
        self.other_tag = os.listdir(data_path)
        self.other_tag.remove(target_tag)
        if '.DS_Store' in self.other_tag:
            self.other_tag.remove('.DS_Store')

    def __getitem__(self, index):
        all_target_file = os.listdir(os.path.join(self.data_path, self.target_tag))
        if '.DS_Store' in all_target_file:
            all_target_file.remove('.DS_Store')
        first_image = all_target_file[random.randint(0, len(all_target_file) - 1)]
        first_image = os.path.join(os.path.join(self.data_path, self.target_tag), first_image)

        is_same = random.randint(0, 1)
        if is_same == 1:
            second_image = all_target_file[random.randint(0, len(all_target_file) - 1)]
            second_image = os.path.join(os.path.join(self.data_path, self.target_tag), second_image)
            is_same = 1.
        else:
            other_tag = self.other_tag[random.randint(0, len(self.other_tag) - 1)]
            all_other_file = os.listdir(os.path.join(self.data_path, other_tag))
            if '.DS_Store' in all_other_file:
                all_other_file.remove('.DS_Store')
            second_image = all_other_file[random.randint(0, len(all_other_file) - 1)]
            second_image = os.path.join(os.path.join(self.data_path, other_tag), second_image)
            is_same = 0.

        img1 = Image.open(first_image)
        img2 = Image.open(second_image)
        img1 = img1.convert('RGB')
        img2 = img2.convert('RGB')

        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        img1 = data_transforms(img1)
        img2 = data_transforms(img2)

        return img1, img2, is_same

    def __len__(self):
        return len(os.listdir(os.path.join(self.data_path, self.target_tag)))


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
            'elapsed_time',
        ]

        self.log_val_headers = [
            'epoch',
            'iteration',
            'train/loss',
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
        acc_all = []

        for batch_idx, (img1, img2, is_same) in tqdm.tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc='Validation Iteration=%d' % self.iteration,
            ncols=80,
            leave=False
        ):
            img1, img2, is_same = Variable(img1), Variable(img2), Variable(is_same)
            is_same = torch.tensor(is_same, dtype=torch.float32)
            with torch.no_grad():
                result = self.model(img1, img2).squeeze(1)

            loss_fn = BCELoss(weight=None, reduce=True)
            loss = loss_fn(result, is_same)
            val_loss += loss

        with open(os.path.join(self.out_path, 'log_val.csv'), 'a') as f:
            elapsed_time = (datetime.datetime.now() - self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration, val_loss, elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        acc = 0.
        is_same = is_same.cpu()
        result = result.cpu()
        for index, item in enumerate(result):
            if item.item() > 0.5 and is_same[index].item() == 1:
                acc += 1
            elif item.item() < 0.5 and is_same[index].item() == 0:
                acc += 1

        acc /= 8
        acc_all.append(acc)

        acc_all = np.array(acc_all)
        print('Val acc: %s' % str(acc_all.mean()))

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
            result = self.model(img1, img2).squeeze(1)

            loss_fn = BCELoss(weight=None, reduce=True)
            loss = loss_fn(result, is_same)
            try:
                loss.backward()
                self.opt.step()
            except Exception as e:
                print(e)

            epoch_loss += loss.detach().numpy()

            if self.iteration >= self.max_iter:
                break

        with open(os.path.join(self.out_path, 'log_train.csv'), 'a') as f:
            elapsed_time = (datetime.datetime.now() - self.timestamp_start).total_seconds()
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
    model = SiameseNetworkSimple()

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
