import tensorflow as tf
import numpy as np
import cv2
import os
import json
import sys

PATH = os.path.abspath('./DataSet')
IMAGE_PATH = os.path.abspath('./DataSet/image')
DICT_PATH = os.path.abspath('./data/char_dict')
TFR_PATH = os.path.abspath('./tfRecord')

train_images = []
train_labels = []
train_imagenames = []

test_images = []
test_labels = []
test_imagenames = []


def build_dataset(info):
    images = np.array([cv2.imread(os.path.join(IMAGE_PATH, tmp), cv2.IMREAD_COLOR) for tmp in info[:, 0]])
    labels = np.array([tmp for tmp in info[:, 1]])
    image_names = np.array([os.path.basename(tmp) for tmp in info[:, 0]])

    return images, labels, image_names


def char_to_int(char):
    temp = ord(char)
    if 65 <= temp <= 90:
        temp = temp + 32

    with open(os.path.join(DICT_PATH, 'ord_map.json'), 'r', encoding='utf8') as json_f:
        res = json.load(json_f)

    for key, value in res.items():
        if value == str(temp):
            temp = int(key)
            break

    return temp


def encode_labels(labels):
    encoded_labeles = []
    lengths = []
    for label in labels:
        encode_label = [char_to_int(char) for char in label]
        encoded_labeles.append(encode_label)
        lengths.append(len(label))
    return encoded_labeles, lengths


def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    value_tmp = []
    is_int = True
    for val in value:
        if not isinstance(val, int):
            is_int = False
            value_tmp.append(int(float(val)))
    if is_int is False:
        value = value_tmp
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def build_tfRecords(images, labels, imagenames, save_dir):
    assert len(images) == len(labels) == len(imagenames)

    labels, length = encode_labels(labels)

    with tf.python_io.TFRecordWriter(save_dir) as writer:
        for index, image in enumerate(images):
            features = tf.train.Features(feature={
                'labels': int64_feature(labels[index]),
                'images': bytes_feature(image),
                'imagenames': bytes_feature(imagenames[index])
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())
            sys.stdout.write('\r>>Writing {:d}/{:d} {:s} tfrecords'.format(index + 1, len(images), imagenames[index]))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        return


def build_info(info):
    # for tmp in info:
    #     line = tmp.strip().split()
    #     print(line[0], line[1])
    return np.array([tmp.strip().split() for tmp in info])


def main():
    train_info = []
    test_info = []
    count = 0
    with open(os.path.abspath(os.path.join(PATH, 'sample.txt')), 'r', encoding='utf8') as file:
        for tmp in file.readlines():
            if count < 10000:
                train_info.append(tmp)
                count += 1
            else:
                test_info.append(tmp)
                count += 1

    train_info = build_info(train_info)
    # test_info = build_info(test_info)

    train_images, train_labels, train_imagenames = build_dataset(train_info)
    # test_images, test_labels, test_imagenames = build_dataset(test_info)

    train_images = [cv2.resize(tmp, (100, 32)) for tmp in train_images]
    train_images = [bytes(list(np.reshape(tmp, 100 * 32 * 3))) for tmp in train_images]
    #
    # test_images = [cv2.resize(tmp, (100, 32)) for tmp in test_images]
    # test_images = [bytes(list(np.reshape(tmp, 100 * 32 * 3))) for tmp in test_images]
    #
    build_tfRecords(train_images, train_labels, train_imagenames, os.path.join(TFR_PATH, 'train/train.tfrecords'))
    # build_tfRecords(test_images, test_labels, test_imagenames, os.path.join(TFR_PATH, 'test'))


if __name__ == '__main__':
    main()
