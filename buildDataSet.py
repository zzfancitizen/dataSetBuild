import tensorflow as tf
import cv2
import numpy as np
import os

IMAGE_PATH = os.path.abspath('./DataSet/image')
GT_PATH = os.path.abspath('./DataSet')
ENGLISH = 'English'


def prepare_dataset(gt_path, image_path, sample_path):
    with open(gt_path, 'r', encoding='utf8') as file1:
        for line in file1.readlines():
            info = np.array(line.strip().split(','))
            if info[1] != ENGLISH:
                os.remove(image_path + os.path.sep + info[0])
            else:
                with open(sample_path, 'a', encoding='utf8') as file2:
                    file2.write(info[0] + ' ' + info[2] + '\n')


def main(gt_path, image_path, sample_path):
    prepare_dataset(gt_path, image_path, sample_path)


def build_tfRecord():
    pass


if __name__ == '__main__':
    gt_path = GT_PATH + os.path.sep + 'gt.txt'
    sample_path = GT_PATH + os.path.sep + 'sample.txt'
    main(gt_path, IMAGE_PATH, sample_path)
