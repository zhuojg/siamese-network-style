import os
import torch
from torchvision import transforms
from siamese_network import SiameseNetwork
from PIL import Image
import random
import matplotlib.pyplot as plt


def test(model_path, data_path, target_tag, pair_num, save_result=False):
    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model_state_dict'])
    result = []

    for i in range(0, pair_num):
        im1, im2, is_same, first_image, second_image = random_data_generator(data_path, target_tag)
        predict = model(im1.unsqueeze(0), im2.unsqueeze(0)).item()
        result.append((first_image, second_image, predict, is_same))

    if save_result:
        plt.figure(figsize=(4, 12), dpi=300)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                            wspace=None, hspace=1.5)

        for i, item in enumerate(result):
            first_image, second_image, predict, is_same = item[0], item[1], item[2], item[3]
            img1 = Image.open(first_image)
            img2 = Image.open(second_image)
            img1 = img1.convert('RGB')
            img2 = img2.convert('RGB')
            plt.subplot(pair_num, 2, 2 * i + 1)
            plt.axis('off')
            plt.title('Label=%s, Prediction=%s' % (str(is_same), str(predict)), fontsize=4)
            plt.imshow(img1)
            plt.subplot(pair_num, 2, 2 * i + 2)
            plt.axis('off')
            plt.title('Real Label=%s' % second_image.split('/')[-2], fontsize=4)
            plt.imshow(img2)

        plt.savefig('./result.png', dpi=300)

    return result


def random_data_generator(data_path, target_tag):
    other_tags = os.listdir(data_path)
    other_tags.remove(target_tag)
    if '.DS_Store' in other_tags:
        other_tags.remove('.DS_Store')

    all_target_file = os.listdir(os.path.join(data_path, target_tag))
    if '.DS_Store' in all_target_file:
        all_target_file.remove('.DS_Store')
    first_image = all_target_file[random.randint(0, len(all_target_file) - 1)]
    first_image = os.path.join(os.path.join(data_path, target_tag), first_image)

    is_same = random.randint(0, 1)
    if is_same == 1:
        second_image = all_target_file[random.randint(0, len(all_target_file) - 1)]
        second_image = os.path.join(os.path.join(data_path, target_tag), second_image)
    else:
        other_tag = other_tags[random.randint(0, len(other_tags) - 1)]
        all_other_file = os.listdir(os.path.join(data_path, other_tag))
        if '.DS_Store' in all_other_file:
            all_other_file.remove('.DS_Store')
        second_image = all_other_file[random.randint(0, len(all_other_file) - 1)]
        second_image = os.path.join(os.path.join(data_path, other_tag), second_image)

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

    return img1, img2, is_same, first_image, second_image


if __name__ == '__main__':
    result = test(model_path='./data/190710_SiameseNetwork.pth.tar',
                  data_path='../style_data_clean',
                  target_tag='modern',
                  pair_num=10,
                  save_result=True)

    print(result)
