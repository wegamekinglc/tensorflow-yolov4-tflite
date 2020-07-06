import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

import datetime as dt
from absl import app, flags
from absl.flags import FLAGS
import os
import shutil
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_string('weights', None, 'pretrained weights')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')


def main(_argv):
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    train_set = Dataset(FLAGS, is_training=True)
    train_set = tf.data.Dataset.from_generator(train_set,
                                               output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                                             tf.float32, tf.float32)).prefetch(
        strategy.num_replicas_in_sync)
    train_set = strategy.experimental_distribute_dataset(train_set)

    logdir = "./data/log"
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = 1

    with strategy.scope():
        input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

        freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)

        feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
        if FLAGS.tiny:
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                if i == 0:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                else:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
        else:
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                if i == 0:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                elif i == 1:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                else:
                    bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)

        model = tf.keras.Model(input_layer, bbox_tensors)

        if FLAGS.weights is None:
            print("Training from scratch")
        else:
            if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
                utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
            else:
                model.load_weights(FLAGS.weights)
            print('Restoring weights from: %s ... ' % FLAGS.weights)

        optimizer = tf.keras.optimizers.Adam()
        if os.path.exists(logdir): shutil.rmtree(logdir)
        writer = tf.summary.create_file_writer(logdir)

        def train_step(image_data, small_target, medium_target, larget_target):
            with tf.GradientTape() as tape:
                pred_result = model(image_data, training=True)
                giou_loss = conf_loss = prob_loss = 0

                # optimizing process
                all_targets = small_target, medium_target, larget_target
                for i in range(len(freeze_layers)):
                    conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                    loss_items = compute_loss(pred, conv, all_targets[i][0], all_targets[i][1],
                                              STRIDES=STRIDES, NUM_CLASS=NUM_CLASS,
                                              IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                    giou_loss += loss_items[0]
                    conf_loss += loss_items[1]
                    prob_loss += loss_items[2]

                total_loss = giou_loss + conf_loss + prob_loss

                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return total_loss

        @tf.function
        def distributed_train_step(image_data, small_target, medium_target, larget_target):
            per_replica_losses = strategy.run(train_step, args=(image_data, small_target, medium_target, larget_target))
            total_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,
                                         axis=None)
            return total_loss

        for epoch in range(first_stage_epochs + second_stage_epochs):
            for image_data, label_sbbox, sbboxes, label_mbbox, mbboxes, label_lbbox, lbboxes in train_set:
                total_loss = distributed_train_step(image_data,
                                                    (label_sbbox, sbboxes),
                                                    (label_mbbox, mbboxes),
                                                    (label_lbbox, lbboxes))
                global_steps += 1
                tf.print("=> STEP %4d   lr: %.6f   total_loss: %4.2f\t" % (global_steps,
                                                                           optimizer.lr.numpy(),
                                                                           total_loss),
                         str(dt.datetime.now()))

                # writing summary data
                with writer.as_default():
                    tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                    tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                writer.flush()

                # for image_data, target in test_set:
                #     test_step(image_data, target)
                if (epoch + 1) % 1 == 0:
                    model.save_weights("./checkpoints/v4/yolov4_{epoch+1}.hf")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
