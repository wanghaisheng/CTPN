import sys
#sys.path.append('/home/ubuntu/github/CTPN/caffe/python')
#sys.path.append('/home/ubuntu/github/tensorflow-alpr/src')

import cv2
import numpy
import tensorflow as tf

import common
import model

import cv2, os, caffe
from cfg import Config as cfg
from other import draw_boxes, resize_im, CaffeModel
from detectors import TextProposalDetector, TextDetector
import os.path as osp
from utils.timer import Timer
from moviepy.editor import VideoFileClip

import collections
import itertools
import math
from tqdm import tqdm
import sys
import os



DEMO_IMAGE_DIR="demo_images/pic_folder"
NET_DEF_FILE="models/deploy.prototxt"
MODEL_FILE="models/ctpn_trained_model.caffemodel"



#NET_DEF_FILE="/home/ubuntu/github/CTPN/models/deploy.prototxt"
#MODEL_FILE="/home/ubuntu/github/CTPN/models/ctpn_trained_model.caffemodel"

if len(sys.argv)>2 and sys.argv[2]=="--no-gpu":
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
    caffe.set_device(cfg.TEST_GPU_ID)

# initialize the detectors
text_proposals_detector=TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
text_detector=TextDetector(text_proposals_detector)

x, y, params = model.get_detect_model()
config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
sess = tf.Session(config=config)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
f = numpy.load("/mnt/DATA02/github/tensorflow-alpr/src/backups/20170327_121800_368350_weights.npz")
param_values = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]


def letter_probs_to_code(letter_probs):
    output = "".join(common.CHARS[i] for i in numpy.argmax(letter_probs, axis=1))
    return output.replace("_", "")


def process_image(im):
    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    text_lines=text_detector.detect(im)
    for box in text_lines:
        left, top, right, bottom = box[:4]
        img_height, img_width = im.shape[0], im.shape[1]
        padding_x = int((right - left) / 4)
        width = padding_x * 2 + (right - left)
        height = (64 * width / 128)
        padding_y = int((height - (bottom - top))/2)
        new_left = int(max(0, left - padding_x))
        new_right = int(min(img_width - 1, right + padding_x))
        new_top = int(max(0, top - padding_y))
        new_bottom = int(min(img_height - 1, bottom + padding_y))

        plate = im[new_top: new_bottom, new_left: new_right]
        plate = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
        plate = cv2.resize(plate, (128, 64))
        im_gray = plate / 255.
        feed_dict = {x: numpy.stack([im_gray])}
        feed_dict.update(dict(zip(params, param_values)))
        y_val = sess.run(y, feed_dict=feed_dict)

        letter_probs = (y_val[0,
                        0,
                        0, 1:].reshape(
            10, len(common.CHARS)))
        letter_probs = common.softmax(letter_probs)

        present_prob = common.sigmoid(y_val[0, 0, 0, 0])
        predicted_label = letter_probs_to_code(letter_probs)
        if present_prob > 0.5:
            print("predict", predicted_label, present_prob)
            cv2.imshow("plate", plate)
            cv2.putText(im, predicted_label, (left, max(0, int(top - 5))), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        cv2.rectangle(im, tuple(box[:2]), tuple(box[2:4]), (0, 255, 0))

    return im

list_image_path = [os.path.join("test", name) for name in os.listdir("test")]
for image_path in list_image_path:
    im = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR)
    output = process_image(im)
    cv2.imshow("output", output)
    cv2.waitKey(0)

