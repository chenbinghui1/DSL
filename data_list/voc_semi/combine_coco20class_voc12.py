import argparse
import numpy as np
import json
import os
import random
import shutil

def combine(args):
        coco = json.load(open(args.cocojson,'r'))
        voc = json.load(open(args.vocjson,'r'))
        with open(os.path.join(args.outjson_path,'voc12_trainval_coco20class.json'),'w') as f:
                res = {}
                res['type'] = voc['type']
                res['categories'] = voc['categories']
                res['images'] = voc['images']
                res['annotations'] = voc['annotations']
                for i in range(len(coco['images'])):
                        coco['images'][i]['file_name'] = coco['images'][i]['file_name'].split('/')[-1]
                        if not os.path.exists(os.path.join(args.cocoimage_path, coco['images'][i]['file_name'])):
                                continue
                        coco['images'][i]['id'] = len(res['images'])
                        res['images'].append(coco['images'][i])
                        shutil.copyfile(os.path.join(args.cocoimage_path, coco['images'][i]['file_name']), os.path.join(args.outimage_path, coco['images'][i]['file_name']))
                with open(os.path.join(args.outtxt_path, 'voc12_trainval_coco20class.txt'),'w') as ff:
                        for i in range(len(res['images'])):
                                ff.write(res['images'][i]['file_name']+'\n')
                json.dump(res,f)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--cocojson', type=str, default="instances_unlabeledtrainval20class.json")
  parser.add_argument('--vocjson', type=str, default="voc12_trainval.json")
  parser.add_argument('--cocoimage_path', type=str, default="/gruntdata1/bhchen/factory/data/semicoco/images/full")
  parser.add_argument('--outtxt_path', type=str, default="/gruntdata1/bhchen/factory/data/semivoc/unlabel_prepared_annos/Industry/")
  parser.add_argument('--outjson_path', type=str, default="")
  parser.add_argument('--outimage_path', type=str, default="/gruntdata1/bhchen/factory/data/semivoc/unlabel_images/full")

  args = parser.parse_args()
  combine(args)
