import os 
import sys
from mmcv import Config
import json
import collections
import argparse
import numpy as np
import cv2
import mmcv
import shutil
import torch
import torch.nn as nn
import scipy.io
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from mmdet.datasets.api_wrappers import COCO

def iou_(a,b):
        area1 = (a[2]-a[0]+1)*(a[3]-a[1]+1)
        area2 = (b[2]-b[0]+1)*(b[3]-b[1]+1)
        if min(a[2],b[2])-max(a[0],b[0])<0 or min(a[3],b[3])-max(a[1],b[1])<0:
                return 0.0
        else:
                inter = (min(a[2],b[2])-max(a[0],b[0])+1)*(min(a[3],b[3])-max(a[1],b[1])+1)
                return float(inter/(area1+area2-inter))

def get_image_list_from_list(list_file):
    coco = COCO(list_file)
    img_ids = coco.get_img_ids()
    image_path_list = []
    for i in img_ids:
        info = coco.load_imgs([i])[0]
        image_path_list.append(info['file_name'])
    num_images = len(image_path_list)
    if num_images == 0:
        print("[ERROR][ModelInfer] Found no image in {}".format(root))
        sys.exit()
    else:
        return image_path_list

def report(args, dst_path):
        iou_thres=0.5
        image_list = get_image_list_from_list(args.gt_list)
        pos_dst = 'pos_imgs/'
        neg_dst = 'neg_imgs/'
        #if not os.path.exists(pos_dst):
        #        os.makedirs(pos_dst)
        #if not os.path.exists(neg_dst):
        #        os.makedirs(neg_dst)
        TP = []
        FP = []
        BG = []
        for idx, name in enumerate(image_list):
                if idx%1000==0:
                        print(idx)
                if os.path.exists(os.path.join(args.pred_path, name+'.json')):
                        with open(os.path.join(args.pred_path, name+'.json'),'r') as f:
                                data = json.load(f)
                        with open(os.path.join(args.gt_path, name+'.json'), 'r') as f:
                                data_gt = json.load(f)
                        if data['targetNum'] == 0:
                                continue

                        #img = Image.open(os.path.join(args.pred_img_root, name))
                        for i in range(data['targetNum']):
                                if data['scores'][i] < args.thres:
                                        continue
                                flag = False
                                tmp = -1
                                for j in range(len(data_gt['tags'])):
                                        if data_gt['tags'][j] != data['tags'][i]:
                                                continue
                                        if iou_(data['rects'][i], data_gt['rects'][j]) < iou_thres:
                                                continue
                                        flag = True
                                        tmp = j
                                        break
                                flag_bg = True
                                for j in range(len(data_gt['tags'])):
                                        if data_gt['tags'][j] != data['tags'][i]:
                                                continue
                                        if iou_(data['rects'][i], data_gt['rects'][j]) == 0.0:
                                                continue
                                        flag_bg = False
                                        break
                                # crop image
                                #img_crop = img.crop((int(data['rects'][i][0]), int(data['rects'][i][1]), int(data['rects'][i][2]), int(data['rects'][i][3])))
                                #if len(np.array(img_crop).shape) == 2:
                                #        a=np.concatenate((np.array(img_crop)[:,:,np.newaxis],np.array(img_crop)[:,:,np.newaxis],np.array(img_crop)[:,:,np.newaxis]), axis=2)
                                #        img_crop = Image.fromarray(a)
                                # create dirs
                                #if not os.path.exists(os.path.join(pos_dst,data['tags'][i])):
                                #        os.makedirs(os.path.join(pos_dst,data['tags'][i]))
                                #if not os.path.exists(os.path.join(neg_dst,data['tags'][i])):
                                #        os.makedirs(os.path.join(neg_dst,data['tags'][i]))
                                # save img_crop
                                if flag == True:
                                        del data_gt['tags'][tmp]
                                        del data_gt['masks'][tmp]
                                        del data_gt['rects'][tmp]
                                        #img_crop.save(os.path.join(pos_dst,data['tags'][i], str(data['scores'][i]) + '_' + name))
                                        TP.append(data['scores'][i])
                                else:
                                        #img_crop.save(os.path.join(neg_dst, data['tags'][i], str(data['scores'][i]) + '_' + name))
                                        if flag_bg == True:
                                            BG.append(data['scores'][i])
                                        else:
                                            FP.append(data['scores'][i])

                else:
                        print(os.path.join(args.pred_path, name+'.json'), " doesn't exists!")
        #TP=np.asarray(TP)
        #FP=np.asarray(FP)
        res={}
        res['TP']=TP
        res['FP']=FP
        res['BG']=BG
        with open('res.json','w') as f:
            json.dump(res,f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="setting input path and output path")
    parser.add_argument('--gt_path',type=str,default=None,help="gt anno path")
    parser.add_argument('--gt_list',type=str,default=None,help="gt list")
    parser.add_argument('--cat_info',type=str,default=None,help="cat infor file")
    parser.add_argument('--pred_path',type=str,default=None,help="eval/per_image_performance")
    parser.add_argument('--pred_image_root',type=str,default="",help="image_root")
    parser.add_argument('--config',type=str,default="",help="config file path")
    parser.add_argument('--pth',type=str,default="",help="det model path")
    parser.add_argument('--thres',type=float,default=0.3,help="threshold of bbox score")
    parser.add_argument('--margin',type=float,default=0.8,help="bbox times")
    args= parser.parse_args()
    dst_path = os.path.abspath(args.pred_path) + '_thres'+str(args.thres)+'_annos_deleteFP-bbox'+str(args.margin)+'/'
    #if not os.path.exists(dst_path):
    #        os.makedirs(dst_path)
    report(args, dst_path)
