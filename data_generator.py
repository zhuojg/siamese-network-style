import os

if __name__ == '__main__':
    data_path = '../style_data_clean'
    target_tag = 'modern'

    other_tag = os.listdir(data_path)
    other_tag.remove(target_tag)
    if '.DS_Store' in other_tag:
        other_tag.remove('.DS_Store')

    all_target_file = os.listdir(os.path.join(data_path, target_tag))

    total_num = len(all_target_file)

    if '.DS_Store' in all_target_file:
        all_target_file.remove('.DS_Store')

    same_pair_num = dif_pair_num = int(total_num / 3 * 0.2)

    used_index = 0

    f = open('modern_pairs_val.txt', 'w')

    target_path = os.path.join(data_path, target_tag)
    for i in range(same_pair_num):
        im1 = os.path.join(target_tag, all_target_file[used_index])
        used_index += 1
        im2 = os.path.join(target_tag, all_target_file[used_index])
        used_index += 1
        f.write('%s,%s,%s\n' % (im1, im2, '1'))

    each_other_tag_num = int(dif_pair_num / len(other_tag))
    for tag in other_tag:
        other_path = os.path.join(data_path, tag)
        other_file = os.listdir(other_path)
        if '.DS_Store' in other_file:
            other_file.remove('.DS_Store')
        for j in range(each_other_tag_num):
            im1 = os.path.join(target_tag, all_target_file[used_index])
            used_index += 1
            im2 = os.path.join(tag, other_file[-j])
            # when generate val data, j has -; while generate train data, j has no -.
            f.write('%s,%s,%s\n' % (im1, im2, '0'))
