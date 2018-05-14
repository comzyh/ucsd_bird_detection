import argparse
import os
import os.path
import random

import tensorflow as tf

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

        filenames.append(filename)
        bboxs.append(bbox)
    yield list(zip(filenames, bboxs))


def main():
    parser = argparse.ArgumentParser(description='Preprocess CUB-200 dataset for qvis')
    parser.add_argument('--datapath', type=str, required=True, help='location of CUB-200 dataset')
    parser.add_argument('--output_path', type=str, required=True, help='location of tf_records')
    parser.add_argument('--train_ratio', type=float, help='ratio of train', default=0.8)

    args = parser.parse_args()

    train_num = 0
    total_num = 0
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with open(os.path.join(args.datapath, 'images.txt')) as images_file:
        with open(os.path.join(args.datapath, 'bounding_boxes.txt')) as bboxs_file:
            image_files = tqdm(images_file.readlines())
            with tf.python_io.TFRecordWriter(os.path.join(args.output_path, 'train')) as train_writer:
                with tf.python_io.TFRecordWriter(os.path.join(args.output_path, 'vaild')) as vaild_writer:
                    for filenames_and_bboxs in get_filenames_and_bboxs(image_files, bboxs_file):
                        train_num += split_train_vaildation(
                            os.path.join(args.datapath, 'images'), filenames_and_bboxs,
                            train_writer, vaild_writer, args.train_ratio)
                        total_num += len(filenames_and_bboxs)


if __name__ == '__main__':
    main()
