import argparse
import os.path

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw

from resnet_model import resnet18_v2

tf.logging.set_verbosity(tf.logging.INFO)


def draw_box(array, box_r=None, box_g=None):

    source_img = Image.fromarray(np.uint8(array))
    draw = ImageDraw.Draw(source_img)
    if box_r is not None:
        draw.rectangle(((box_r[0], box_r[1]), (box_r[0] + box_r[2], box_r[1] + box_r[3])), outline="red")
    if box_g is not None:
        draw.rectangle(((box_g[0], box_g[1]), (box_g[0] + box_g[2], box_g[1] + box_g[3])), outline="red")
    return source_img


def get_dataset(tfrecord_dir, setname):
    DATASET_SIZES = {
        'train': 9422,
        'validation': 2366,
    }

    # filenames = tf.placeholder(tf.string, shape=[None])
    filenames = [os.path.join(tfrecord_dir, setname)]
    dataset = tf.data.TFRecordDataset(filenames)

    def parser(record):
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/bbox': tf.FixedLenFeature((4, ), tf.float32, default_value=tf.zeros(4, dtype=tf.float32)),
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.image.decode_jpeg(parsed["image/encoded"], channels=3)
        bbox = parsed['image/bbox']
        target_shape = (224, 224, 3)  # (W, H)

        xscale = target_shape[0] / tf.to_float(tf.shape(image)[1])  # image shape is (H, W)
        yscale = target_shape[1] / tf.to_float(tf.shape(image)[0])

        x = bbox[0] * xscale  # / 112.0 - 1.0
        y = bbox[1] * yscale  # / 112.0 - 1.0
        w = bbox[2] * xscale  # / 112.0
        h = bbox[3] * yscale  # / 112.0

        bbox = tf.stack([x, y, w, h], axis=0)
        image = tf.image.resize_images(image, target_shape[:2])

        means = [122.81981232, 127.39550182, 108.8589721]
        channels = tf.split(axis=2, num_or_size_splits=3, value=image)
        for i in range(3):
            channels[i] -= means[i]
        image = tf.concat(axis=2, values=channels)

        return image, bbox

    dataset = dataset.map(parser, num_parallel_calls=8)
    return dataset


def box_intersection(box1, box2):
    def ltrb(box):
        return tf.stack([box[:, 0], box[:, 1], box[:, 0] + tf.maximum(box[:, 2], 0), box[:, 1] + tf.maximum(box[:, 3], 0)], axis=-1)

    box1 = ltrb(box1)
    box2 = ltrb(box2)
    l = tf.maximum(box1[:, 0], box2[:, 0])
    t = tf.maximum(box1[:, 1], box2[:, 1])
    r = tf.minimum(box1[:, 2], box2[:, 2])
    b = tf.minimum(box1[:, 3], box2[:, 3])

    v1 = tf.maximum(box1[:, 2] - box1[:, 0], 0) * tf.maximum(box1[:, 3] - box1[:, 1], 0)
    v2 = tf.maximum(box2[:, 2] - box2[:, 0], 0) * tf.maximum(box2[:, 3] - box2[:, 1], 0)
    vin = tf.maximum(r - l, 0) * tf.maximum(b - t, 0)
    vall = v1 + v2 - vin
    score = tf.where(tf.greater(vin, 0), vin / vall, tf.zeros_like(vin))

    return score


def model_fn(features, labels, mode):

    x = resnet18_v2(inputs=features, N_final=4, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

    loss = tf.losses.huber_loss(labels=labels, predictions=x)
    score = box_intersection(x, labels)

    correct = tf.greater(score, 0.75)
    correct_ref = tf.ones_like(correct)
    accuracy= tf.metrics.accuracy(labels=correct_ref, predictions=correct, name='accuracy')
    ioup5 = tf.metrics.accuracy(labels=correct_ref, predictions=tf.greater(score, 0.5), name='accuracy')
    mean_score = tf.metrics.mean(score)

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(learning_rate=0.0005,
                                                   global_step=tf.train.get_global_step(),
                                                   decay_steps=1000,
                                                   decay_rate=0.5,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam')

        train_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for batch_normalization
        train_op.append(optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()))
        train_op.append(accuracy[1])
        train_op.append(mean_score[1])
        train_op.append(ioup5[1])
        train_op = tf.group(*train_op)

        tf.summary.scalar('acc', accuracy[0])
        tf.summary.scalar('mean_score', mean_score[0])
        tf.summary.scalar('ioup5', ioup5[0])
        tf.summary.scalar('learning_rate', learning_rate)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "acc": accuracy,
        "mean_score": mean_score,
        "ioup5": ioup5,
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    parser = argparse.ArgumentParser(description='train CUB-200 detection')
    parser.add_argument('--datapath', type=str, required=True, help='location of CUB-200 tfrecords')
    # parser.add_argument('--model', type=str, default='/tmp/cub_200_resnet', help='location of model')
    args = parser.parse_args()
    config = tf.estimator.RunConfig(model_dir="/tmp/ucsdbird",
                                    save_summary_steps=10,
                                    save_checkpoints_steps=100,
                                    keep_checkpoint_max=10,
                                    log_step_count_steps=10)

    ucsd_bird_detector = tf.estimator.Estimator(model_fn=model_fn, config=config)

    batch_size = 100

    def input_fn_factory(setname):
        def input_fn():
            dataset = get_dataset(args.datapath, setname)
            dataset = dataset.shuffle(1000).batch(batch_size).prefetch(2)
            iterator = dataset.make_one_shot_iterator()
            image_batch, label_batch = iterator.get_next()
            return image_batch, label_batch

        return input_fn

    for epoch in range(40):
        print('Epoch {}'.format(epoch))
        ucsd_bird_detector.train(input_fn=input_fn_factory('train'))
        eval_results = ucsd_bird_detector.evaluate(input_fn=input_fn_factory('validation'))
        print(eval_results)


if __name__ == '__main__':
    main()
