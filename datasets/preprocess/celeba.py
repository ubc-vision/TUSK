import os
import shutil
from tqdm import tqdm
import matplotlib.image as mpimg

def main():
    celeba_root = '/ubc/cs/research/kmyi/datasets/CelebA'
    anno_folder = 'Anno'
    eval_folder = 'Eval'
    img_folder = 'img_celeba'
    
    # create training set for self suoervised training
    with open(os.path.join(celeba_root,eval_folder,'list_eval_partition.txt'), 'r') as c_f, \
        open(os.path.join(celeba_root,eval_folder,'testing.txt'), 'r') as m_test_f, \
        open(os.path.join(celeba_root,eval_folder,'CelebA_train.txt'), 'w') as c_train_f:
        c_lines = c_f.read().splitlines()
        m_test_lines = m_test_f.read().splitlines()
        c_train_lines = []
        for line in c_lines:
            str_list = line.split(' ')
            if str_list[1] == '0':
                c_train_lines.append(str_list[0])
        for line in m_test_lines:
            if line in c_train_lines:
                c_train_lines.remove(line)
        for line in c_train_lines:
            c_train_f.write(line+'\n')

    # rename MAFL training and testing txt
    shutil.copyfile(os.path.join(celeba_root,eval_folder,'testing.txt'), os.path.join(celeba_root,eval_folder,'MAFL_test.txt'))
    shutil.copyfile(os.path.join(celeba_root,eval_folder,'training.txt'), os.path.join(celeba_root,eval_folder,'MAFL_train.txt'))

    # create image size list file
    img_path = os.path.join(celeba_root,img_folder)
    img_names = [f for f in os.listdir(img_path) if f.endswith('jpg')]
    img_names.sort()
    with open(os.path.join(celeba_root,anno_folder, 'list_imsize_celeba.txt'), 'w') as f:
        for img_name in tqdm(img_names):
            img_size = mpimg.imread(os.path.join(img_path,img_name)).shape
            f.write('{} {} {}\n'.format(img_name,img_size[0],img_size[1]))
            
if __name__ == '__main__':
    main()
    