import time
from pathlib import Path
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, YOLOv4_tiny, decode
from PIL import Image
from core.config import cfg
import cv2
import numpy as np
import tensorflow as tf

flags.DEFINE_string('framework', 'tf', '(tf, tflite')
flags.DEFINE_string('weights', '/data/dev/cheng/remote/tf2-yolov4/checkpoints/v4/yolov4_80.hf/yolov4_80.hf',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('output', './data/results', 'path to output image')


class Detector:

    def __init__(self):
        self.images = None
        self._load_model()

    def load_images(self, path):
        path = Path(path)
        files = [file.as_posix() for file in path.glob("*.jpg")]
        self.images = {image_path: self._load_image(image_path) for image_path in files}

    def _load_image(self, image_path):
        input_size = FLAGS.size
        image_name = Path(image_path).name
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]

        image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        return image_name, image_data, original_image, original_image_size

    def _load_model(self):
        NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
        input_size = FLAGS.size

        if FLAGS.framework == 'tf':
            input_layer = tf.keras.layers.Input([input_size, input_size, 3])
            if FLAGS.tiny:
                if FLAGS.model == 'yolov3':
                    feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
                else:
                    feature_maps = YOLOv4_tiny(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
                model.summary()
                utils.load_weights_tiny(model, FLAGS.weights, FLAGS.model)
            else:
                if FLAGS.model == 'yolov3':
                    feature_maps = YOLOv3(input_layer, NUM_CLASS)
                    bbox_tensors = []
                    for i, fm in enumerate(feature_maps):
                        bbox_tensor = decode(fm, NUM_CLASS, i)
                        bbox_tensors.append(bbox_tensor)
                    model = tf.keras.Model(input_layer, bbox_tensors)
                    utils.load_weights_v3(model, FLAGS.weights)
                elif FLAGS.model == 'yolov4':
                    feature_maps = YOLOv4(input_layer, NUM_CLASS)
                    bbox_tensors = []
                    for i, fm in enumerate(feature_maps):
                        bbox_tensor = decode(fm, NUM_CLASS, i)
                        bbox_tensors.append(bbox_tensor)
                    model = tf.keras.Model(input_layer, bbox_tensors)

                    if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
                        utils.load_weights(model, FLAGS.weights)
                    else:
                        model.load_weights(FLAGS.weights)
            model.summary()
        else:
            # Load TFLite model and allocate tensors.
            model = tf.lite.Interpreter(model_path=FLAGS.weights)
        self.model = model

    def _process_one_image(self, image_name, image_data, original_image, original_image_size):
        input_size = FLAGS.size
        if FLAGS.tiny:
            STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
            XYSCALE = cfg.YOLO.XYSCALE_TINY
        else:
            STRIDES = np.array(cfg.YOLO.STRIDES)
            if FLAGS.model == 'yolov4':
                ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
            else:
                ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
            XYSCALE = cfg.YOLO.XYSCALE

        if FLAGS.framework == 'tf':
            pred_bbox = self.model.predict(image_data)
        else:
            self.model.allocate_tensors()
            # Get input and output tensors.
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            self.model.set_tensor(input_details[0]['index'], image_data)
            self.model.invoke()
            pred_bbox = [self.model.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        if FLAGS.model == 'yolov4':
            if FLAGS.tiny:
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
            else:
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
        else:
            pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
        bboxes = utils.nms(bboxes, 0.213, method='nms')

        image = utils.draw_bbox(original_image, bboxes)
        image = Image.fromarray(image)
        # image.show()
        file_path = Path(FLAGS.output) / image_name
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(file_path.as_posix(), image)
        logging.info(f"save to {file_path}")
        return bboxes

    def detect_all(self):
        results = dict()
        for k, packet in self.images.items():
            results[k] = self._process_one_image(*packet)
        return results


def main(_argv):
    paths = ["/data/dev/cheng/heads_data/20200507_mozi/0507-0509image_sample",
             "/data/dev/cheng/heads_data/20200521_coolrat/0521image_sample"]
    detector = Detector()

    for path in paths:
        detector.load_images(path)
        results = detector.detect_all()
        print(results)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
