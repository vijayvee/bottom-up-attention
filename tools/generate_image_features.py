#!/usr/bin/env python

"""Generate bottom-up attention features as an h5 file.
Code borrowed from https://github.com/peteanderson80/bottom-up-attention.
The original work on bottom-up-attention for VQA can be found here: https://arxiv.org/abs/1707.07998"""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
import glob
from tqdm import tqdm
import matplotlib
#matplotlib.use('Agg')
import h5py

from utils.timer import Timer
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import scipy.io as sio
import argparse

import caffe
import argparse
import pprint
import time, os, sys
import base64
import numpy as np
import cv2
import csv
from multiprocessing import Process
import random
import json

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
eps = 1e-8
CLASSES = ['__background__']
with open(os.path.join(cfg.DATA_DIR, 'vg/objects_vocab.txt')) as f:
  for object in f.readlines():
    CLASSES.append(object.lower().strip())

ATTRS = []
with open(os.path.join(cfg.DATA_DIR, 'vg/attributes_vocab.txt')) as f:
  for attr in f.readlines():
    ATTRS.append(attr.lower().strip())

RELATIONS = []
with open(os.path.join(cfg.DATA_DIR, 'vg/relations_vocab.txt')) as f:
  for rel in f.readlines():
    RELATIONS.append(rel.lower().strip())


def vis_detections(ax, class_name, dets, attributes, rel_argmax, rel_score, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, 4]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )

        if attributes is not None:
            att = np.argmax(attributes[i])
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f} ({:s})'.format(class_name, score, ATTRS[att]),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')
        else:
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

        #print class_name
        #print 'Outgoing relation: %s' % RELATIONS[np.argmax(rel_score[i])]

    ax.set_title(('detections with '
                  'p(object | box) >= {:.1f}').format(thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2, visualize=False):
    MIN_BOXES, MAX_BOXES=36,36
    NMS_THRESH = 0.05
    CONF_THRESH = 0.1
    ATTR_THRESH = 0.1
    im = cv2.imread(im_file)
#    import ipdb; ipdb.set_trace()
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)
    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    pool5 = net.blobs['pool5_flat'].data
    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
    #import ipdb; ipdb.set_trace()
    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]

    #Normalize scores of best detections, sum their features to
    #obtain bottom-up attention features of im

    best_scores = max_conf[keep_boxes]
    best_feats = pool5[keep_boxes]
    scores_norm = np.expand_dims(np.exp(best_scores)/np.sum(np.exp(best_scores))+eps,axis=1)
    cumulative_feats = scores_norm.T.dot(best_feats)
    sum_feats = np.sum(best_feats,axis=0)
    #print np.mean(cumulative_feats),np.mean(best_feats)
    #import ipdb; ipdb.set_trace()
    if visualize:
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            if attr_scores is not None:
                attributes = attr_scores[keep]
            else:
                attributes = None
            if rel_scores is not None:
                rel_argmax_c = rel_argmax[keep]
                rel_score_c = rel_score[keep]
            else:
                rel_argmax_c = None
                rel_score_c = None
            vis_detections(ax, cls, dets, attributes, rel_argmax_c, rel_score_c, thresh=CONF_THRESH)
        plt.savefig('./'+im_file.split('/')[-1].replace(".jpg", "_demo.png"))


    return {
        'image_id': image_id,
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes' : len(keep_boxes),
        'boxes': base64.b64encode(cls_boxes[keep_boxes]),
        'features': base64.b64encode(pool5[keep_boxes]),
        'cumulative_feats':cumulative_feats,
        'sum_feats':sum_feats
    }

def main():
    caffe.set_mode_gpu()
    caffe.set_device(0)
    prototxt = '/home/youssef/visdial/data/models/faster_rcnn/test.prototxt' #Set to location of test/deploy prototxt
    weights = '/home/youssef/visdial/data/models/faster_rcnn/resnet101_faster_rcnn_final.caffemodel' #Set to location of caffemodel
    image_root = '/media/data_cifs/image_datasets/coco_2014/coco_images'
    #image_file = '/media/data_cifs/image_datasets/coco_2014/coco_images/train2014/COCO_train2014_000000000025.jpg' #Set to path of image file
    net = caffe.Net(prototxt, caffe.TEST, weights=weights)
    image_id = 0
    cumulative_feats = np.zeros((1,2048))
    sum_feats = np.zeros((1,2048))
    visdial_data = json.load(open('/home/youssef/visdial/data/visdial_params.json'))
    train_ids = visdial_data['unique_img_train']
    for i in tqdm(range(len(train_ids)),desc="Extracting Faster-RCNN features on COCO..."):
        im = '%s/train2014/COCO_train2014_%012d.jpg'%(image_root,train_ids[i])
        detections = get_detections_from_im(net,im,image_id+i)
        #import ipdb; ipdb.set_trace()
        cumulative_feats = np.vstack([cumulative_feats,detections['cumulative_feats']])
        sum_feats = np.vstack([sum_feats,detections['sum_feats']])
    cumulative_feats, sum_feats = cumulative_feats[1:], sum_feats[1:]
    cumulative_h5 = h5py.File('data_cumulative_train.h5', 'w')
    sum_h5 = h5py.File('data_sum_train.h5', 'w')
    cumulative_h5.create_dataset('/images_train',data=cumulative_feats)
    sum_h5.create_dataset('/images_train',data=sum_feats)
    cumulative_h5.close()
    sum_h5.close()
if __name__ == '__main__':
    main()
