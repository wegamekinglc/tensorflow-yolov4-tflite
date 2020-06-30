from absl import app, flags
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', None, 'pretrained weights')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
        NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
        STRIDES = np.array(cfg.YOLO.STRIDES)
        IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
        XYSCALE = cfg.YOLO.XYSCALE
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)

        freeze_layouts = ['conv2d_93', 'conv2d_101', 'conv2d_109']

        if FLAGS.tiny:
            freeze_layouts = ['conv2d_9', 'conv2d_12']
            STRIDES = np.array(cfg.YOLO.STRIDES_TINY)

            feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode_train(fm, NUM_CLASS, STRIDES, ANCHORS, i)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
        else:
            if FLAGS.model == 'yolov3':
                feature_maps = YOLOv3(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode_train(fm, NUM_CLASS, STRIDES, ANCHORS, i)
                    bbox_tensors.append(fm)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
            elif FLAGS.model == 'yolov4':
                feature_maps = YOLOv4(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode_train(fm, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                    bbox_tensors.append(fm)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)

        if FLAGS.weights == None:
            print("Training from scratch")
        else:
            if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
                if FLAGS.tiny:
                    utils.load_weights_tiny(model, FLAGS.weights)
                else:
                    if FLAGS.model == 'yolov3':
                        utils.load_weights_v3(model, FLAGS.weights)
                    else:
                        utils.load_weights(model, FLAGS.weights)
            else:
                model.load_weights(FLAGS.weights)
            print('Restoring weights from: %s ... ' % FLAGS.weights)
        optimizer = tf.keras.optimizers.Adam()

    trainset = Dataset(is_training=True, tiny=FLAGS.tiny)

    logdir = "./data/log"
    isfreeze = False
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)

    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(len(freeze_layouts)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS,
                                          IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return total_loss

    @tf.function
    def distributed_train_step(image_data, target):
        per_replica_losses = strategy.run(train_step, args=(image_data, target))
        total_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                     axis=None)
        global_steps.assign_add(1)
        return total_loss

    for epoch in range(first_stage_epochs + second_stage_epochs):
        if epoch < first_stage_epochs:
            if not isfreeze:
                isfreeze = True
                for name in freeze_layouts:
                    freeze = model.get_layer(name)
                    freeze_all(freeze)
        elif epoch >= first_stage_epochs:
            if isfreeze:
                isfreeze = False
                for name in freeze_layouts:
                    freeze = model.get_layer(name)
                    unfreeze_all(freeze)
        for image_data, target in trainset:
            total_loss = distributed_train_step(image_data, target)
            tf.print("=> STEP %4d   lr: %.6f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(), total_loss))

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            writer.flush()

        if (epoch + 1) % 20 == 0:
            file_name = f"./checkpoints/yolov4_{epoch}.h5"
            model.save_weights(file_name)
            print(f"epoch {epoch + 1} saved at: {file_name}")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
