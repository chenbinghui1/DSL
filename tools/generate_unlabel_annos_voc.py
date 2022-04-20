import json
import os
import argparse
from mmdet.datasets.api_wrappers import COCO

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']

def report(args, dst_path):
    coco = COCO(args.input_list)
    cat_ids = coco.get_cat_ids(cat_names=CLASSES)
    cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}

    label2word = json.load(open(args.cat_info,'r'))['id2cat']
    cnt=0
    D = json.load(open(args.input_list,'r'))
    with open(args.input_path,'r') as f:
        data = json.load(f)
        #{'image_id': 391895, bbox': [346.72308349609375, 19.586660385131836, 125.62673950195312, 280.1312656402588], 'score': 0.8345049619674683, 'category_id': 1}
        for i in range(len(data)):
            name = D['images'][data[i]['image_id']]['file_name']+'.json'
            if not os.path.exists(os.path.join(dst_path, name)):
                with open(os.path.join(dst_path, name),'w') as F:
                    tmp_data={}
                    tmp_data["imageName"] = "full/"+D['images'][data[i]['image_id']]['file_name']
                    tmp_data["targetNum"] = 0
                    tmp_data["rects"] = []
                    tmp_data["tags"] = []
                    tmp_data["scores"] = []
                    tmp_data["masks"] = []
                    json.dump(tmp_data,F,ensure_ascii=False,indent=4)
            if float(data[i]['score']) <= float(args.thres):
                continue
            cnt +=1
            if cnt%10000 ==0:
                print(cnt)
            if os.path.exists(os.path.join(dst_path, name)):
                with open(os.path.join(dst_path, name),'r') as F:
                    tmp_data = json.load(F)
                tmp_data["targetNum"] = tmp_data["targetNum"] + 1
                tmp_data["rects"].append([data[i]['bbox'][0], data[i]['bbox'][1], data[i]['bbox'][0]+data[i]['bbox'][2], data[i]['bbox'][1]+data[i]['bbox'][3]])
                tmp_data["tags"].append(label2word[str(cat2label[data[i]['category_id']])])
                tmp_data["scores"].append(data[i]['score'])
                tmp_data["masks"].append([])
                with open(os.path.join(dst_path,name),'w') as F:
                    json.dump(tmp_data,F,ensure_ascii=False,indent=4)
    print("total num : ",cnt)

    cnt = 0
    with open(args.input_list,'r') as f:
        data=json.load(f)
        for i in range(len(data['images'])):
            if not os.path.exists(os.path.join(dst_path, data['images'][i]['file_name']+'.json')):
                with open(os.path.join(dst_path, data['images'][i]['file_name']+'.json'),'w') as F:
                    cnt +=1
                    tmp_data={}
                    tmp_data["imageName"] = "full/"+data['images'][i]['file_name']
                    tmp_data["targetNum"] = 0
                    tmp_data["rects"] = []
                    tmp_data["tags"] = []
                    tmp_data["scores"] = []
                    tmp_data["masks"] = []
                    json.dump(tmp_data,F,ensure_ascii=False,indent=4)
    print("number of images with no pseudo-labels: ", cnt)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="setting input path and output path")
    parser.add_argument('--input_path',type=str,default=None,help="path to XX-unlabel.bbox.json")
    parser.add_argument('--input_list',type=str,default=None,help="path to data list: data_list/voc_semi/voc12_trainval.json")
    parser.add_argument('--cat_info',type=str,default=None,help="path to category_info: /gruntdata1/bhchen/factory/data/semivoc/mmdet_category_info.json")
    parser.add_argument('--thres',type=float,default=0.1,help="threshold: 0.1 is used as default")
    args= parser.parse_args()
    dst_path = os.path.abspath(args.input_path) + '_thres'+str(args.thres)+'_annos/'
    if not os.path.exists(dst_path):
            os.makedirs(dst_path)
    report(args, dst_path)
