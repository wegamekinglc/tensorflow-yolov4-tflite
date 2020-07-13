import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

import os
from pathlib import Path
import cv2
import numpy as np
import core.utils as utils
from absl import app, flags
from absl.flags import FLAGS
from core.yolov4 import filter_boxes
from PIL import Image
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', '/data/dev/cheng/remote/tf2-yolov4/models/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('output', 'data/results', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')


def _load_model():
    if FLAGS.framework == 'tflite':
        infer = tf.lite.Interpreter(model_path=FLAGS.weights)
        infer.allocate_tensors()
        input_details = infer.get_input_details()
        output_details = infer.get_output_details()
        print(input_details)
        print(output_details)
    else:
        infer = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    return infer


def _pred(infer, file_name, original_image, images_data, input_details=None, output_details=None):
    input_size = FLAGS.size
    if FLAGS.framework == 'tflite':
        infer.set_tensor(input_details[0]['index'], images_data)
        infer.invoke()
        pred = [infer.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        if FLAGS.model == 'yolov3' and FLAGS.tiny:
            boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                            input_shape=tf.constant([input_size, input_size]))
        else:
            boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                            input_shape=tf.constant([input_size, input_size]))
    else:
        infer = infer.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image = utils.draw_bbox(original_image, pred_bbox)
    # image = utils.draw_bbox(image_data*255, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    # image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(FLAGS.output, file_name), image)


def _generate_data(file_path):
    input_size = FLAGS.size
    file_name = Path(file_path).name

    original_image = cv2.imread(file_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    # images_data = image_data[np.newaxis, ...].astype(np.float32)

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)
    return file_name, original_image, images_data


def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    infer = _load_model()

    source_roots = ["/data/dev/cheng/heads_data/20200507_mozi/0507-0509image_sample",
                    "/data/dev/cheng/heads_data/20200521_coolrat/0521image_sample"]
    for source_root in source_roots:
        img_files = Path(source_root).glob("*.jpg")
        for file in img_files:
            file_name, original_image, images_data = _generate_data(file.as_posix())
            _pred(infer, file_name, original_image, images_data)
            print(file_name)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
