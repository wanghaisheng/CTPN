#
# The codes are used for implementing CTPN for scene text detection, described in: 
#
# Z. Tian, W. Huang, T. He, P. He and Y. Qiao: Detecting Text in Natural Image with
# Connectionist Text Proposal Network, ECCV, 2016.
#
# Online demo is available at: textdet.com
# 
# These demo codes (with our trained model) are for text-line detection (without 
# side-refiement part).  
#
#
# ====== Copyright by Zhi Tian, Weilin Huang, Tong He, Pan He and Yu Qiao==========

#            Email: zhi.tian@siat.ac.cn; wl.huang@siat.ac.cn
# 
#   Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences
#
#

from cfg import Config as cfg
from other import draw_boxes, resize_im, CaffeModel
import cv2, os, caffe, sys
from detectors import TextProposalDetector, TextDetector
import os.path as osp
from utils.timer import Timer

DEMO_IMAGE_DIR="demo_images/pic_folder"
NET_DEF_FILE="models/deploy.prototxt"
MODEL_FILE="models/ctpn_trained_model.caffemodel"

if len(sys.argv)>1 and sys.argv[1]=="--no-gpu":
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
    caffe.set_device(cfg.TEST_GPU_ID)

# initialize the detectors
text_proposals_detector=TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
text_detector=TextDetector(text_proposals_detector)



### 能够运行


from glob import glob
timer=Timer()
for im_name in glob("demo_images/pic_folder/*.jpg"):
    print im_name
    im_file=osp.join(im_name)
    im=cv2.imread(im_file)
    timer.tic()
    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    text_lines,text_proposals,scores=text_detector.detect(im)
#    text_lines=enlarge_boxes(text_lines)
    print "Time: %f"%timer.toc()
    print text_lines
    im_with_text_lines=draw_boxes(im, text_lines, caption=im_name, is_display=False)
    display(im_with_text_lines)
    box_count = 0
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

        crop_img = im[new_top: new_bottom, new_left: new_right]
        #crop_img = cv2.cvtColor(crop_img,cv2.COLOR_RGB2GRAY)
        crop_img = cv2.resize(crop_img, (300,300))
        cv2.imwrite('demo_images/pic_folder/box_text_proposals_{}.jpg'.format(box_count),crop_img)
        box_count += 1     


print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "Thank you for trying our demo. Press any key to exit..."
cv2.waitKey(0)





###
from glob import glob
timer=Timer()
for im_name in glob("demo_images/pic_folder/*.jpg"):
    print im_name
    im_file=osp.join(im_name)
    im=cv2.imread(im_file)
    timer.tic()
    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    text_lines,text_proposals,scores=text_detector.detect(im)
#    text_lines=enlarge_boxes(text_lines)
    print "Time: %f"%timer.toc()
    print text_lines
    im_with_text_lines=draw_boxes(im, text_lines, caption=im_name, is_display=False)
    box_count = 0
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

        crop_img = im[new_top: new_bottom, new_left: new_right]
        #crop_img = cv2.cvtColor(crop_img,cv2.COLOR_RGB2GRAY)
        crop_img = cv2.resize(crop_img, (300,300))
        cv2.imwrite('demo_images/pic_folder/box_text_proposals_{}.jpg'.format(box_count),crop_img)
        box_count += 1     


print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
print "Thank you for trying our demo. Press any key to exit..."
cv2.waitKey(0)





from glob import glob
timer=Timer()
for im_name in glob("demo_images/*.jpg"):
    print im_name
    im_file=osp.join(im_name)
    im=cv2.imread(im_file)
    timer.tic()
    im, f=resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
    text_lines,text_proposals,scores=text_detector.detect(im)
#    text_lines=enlarge_boxes(text_lines)
    print "Time: %f"%timer.toc()
    print text_lines
    im_with_text_lines=draw_boxes(im, text_lines, caption=im_name, is_display=False)
    cv2.imshow(im_name,im_with_text_lines)
    box_count = 0
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

        crop_img = im[new_top: new_bottom, new_left: new_right]
        #crop_img = cv2.cvtColor(crop_img,cv2.COLOR_RGB2GRAY)
        crop_img = cv2.resize(crop_img, (300,300))
        cv2.imwrite('demo_images/pic_folder/'+im_name+'_box_text_proposals_{}.jpg'.format(box_count),crop_img)
        box_count += 1     



        
