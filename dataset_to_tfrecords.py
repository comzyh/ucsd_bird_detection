import argparse
import os
import os.path
import random

import tensorflow as tf
import numpy as np

from tqdm import tqdm


def split_train_vaildation(images_path, filenames_and_bboxs, train_writer, vaild_writer, train_ratio=0.8):
    indexs = list(range(len(filenames_and_bboxs)))
    random.shuffle(indexs)
    for i in range(len(filenames_and_bboxs)):
        filename, bbox = filenames_and_bboxs[indexs[i]]
        with open(os.path.join(images_path, filename), 'rb') as f:
            record = tf.train.Example(features=tf.train.Features(feature={
                "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[f.read()])),
                "image/bboxs": tf.train.Feature(float_list=tf.train.FloatList(value=bbox)),
            }))
        if i <= round(len(filenames_and_bboxs) * train_ratio):
            train_writer.write(record.SerializeToString())
        else:
            vaild_writer.write(record.SerializeToString())

    train_num = round(len(filenames_and_bboxs) * train_ratio)
    return train_num


def get_filenames_and_bboxs(images, bounding_boxes):
    filenames = []
    bboxs = []
    for line in images:
        if not line:
            break
        _, filename = line.split()
        bbox = list(map(float, bounding_boxes.readline().split()[1:]))
        if len(filenames) != 0 and filename[:3] != filenames[-1][:3]:
            yield list(zip(filenames, bboxs))
            filenames = []
            bboxs = []

        filenames.append(filename)
        bboxs.append(bbox)
    yield list(zip(filenames, bboxs))


def get_channel_average(datapath):
    images_filepath = os.path.join(datapath, 'images.txt')
    with open(images_filepath) as f:
        records = sum(map(lambda x: x != '', f.readlines()))

    dataset = tf.data.TextLineDataset([images_filepath])

    def _read_image(line):
        filename = tf.py_func(lambda x: os.path.join(datapath, 'images', x.split()[1].decode()), [line], tf.string)
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_image(image_string)
        mean = tf.reduce_mean(tf.to_float(image_decoded), [0, 1])  # reduce H and W
        return mean

    dataset = dataset.map(_read_image)
    iterator = dataset.make_one_shot_iterator()
    next_ele = iterator.get_next()
    sess = tf.Session()

    avg = np.zeros(3)
    count = 0
    for i in tqdm(range(records)):
        try:
            avg += sess.run(next_ele)
            count += 1
        except tf.errors.OutOfRangeError:
            break

    return avg / count
    # [122.81981232 127.39550182 108.8589721 ]


def main():
    parser = argparse.ArgumentParser(description='Preprocess CUB-200 dataset for qvis')
    parser.add_argument('--datapath', type=str, required=True, help='location of CUB-200 dataset')
    parser.add_argument('--output_path', type=str, help='location of tf_records')
    parser.add_argument('--train_ratio', type=float, help='ratio of train', default=0.8)
    parser.add_argument('--get_channel_average', action='store_true')

    args = parser.parse_args()

    if args.get_channel_average:
        avg = get_channel_average(args.datapath)
        print(avg)
        return

    train_num = 0
    total_num = 0
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with open(os.path.join(args.datapath, 'images.txt')) as images_file:
        with open(os.path.join(args.datapath, 'bounding_boxes.txt')) as bboxs_file:
            image_files = tqdm(images_file.readlines())
            with tf.python_io.TFRecordWriter(os.path.join(args.output_path, 'train')) as train_writer:
                with tf.python_io.TFRecordWriter(os.path.join(args.output_path, 'vaild')) as vaild_writer:
                    random.seed(515)  # deadline of this homework
                    for filenames_and_bboxs in get_filenames_and_bboxs(image_files, bboxs_file):
                        train_num += split_train_vaildation(
                            os.path.join(args.datapath, 'images'), filenames_and_bboxs,
                            train_writer, vaild_writer, args.train_ratio)
                        total_num += len(filenames_and_bboxs)
                    print("{}/{} training data, {}".format(train_num, total_num, train_num / total_num))


if __name__ == '__main__':
    main()
