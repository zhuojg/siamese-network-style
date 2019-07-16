import os
from binary_classification import SiameseNetworkClassifier
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import random
import matplotlib.pyplot as plt


def random_data_generator(num):
    result = []
    f = open('./all.txt', 'r')

    data = f.readlines()
    random.shuffle(data)

    for i, line in enumerate(data):
        if i >= num:
            break
        result.append(line[:-1])

    return result


def test(pre_train_model_path):
    model = SiameseNetworkClassifier('')
    model.load_state_dict(torch.load(pre_train_model_path, map_location='cpu')['model_state_dict'])

    sample_num = 50

    data = random_data_generator(sample_num)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    plt.figure(figsize=(2, 36), dpi=300)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=1.5)

    for i, img in enumerate(data):
        im = Image.open(os.path.join('../style_data_clean', img))
        im = im.convert('RGB')
        plt.subplot(sample_num, 1, i + 1)
        plt.axis('off')
        plt.imshow(im)
        predict = model(data_transforms(im).unsqueeze(0))
        plt.title('Label=%s, Possibility of Modern=%s' % (img.split('/')[0], str(predict.item())), fontsize=4)

    plt.savefig('./result.png', dpi=300)


if __name__ == '__main__':
    test('./data/190716_binary_classifier_modern.pth.tar')
