import json
import os
import argparse
import shutil

def convert(args):
        print("Generating DSL-style data dir")
        if not os.path.exists(args.output):
                os.mkdir(args.output)

        if not os.path.exists(os.path.join(args.output,'prepared_annos/Industry/annotations/full')):
                os.makedirs(os.path.join(args.output,'prepared_annos/Industry/annotations/full'))
        if not os.path.exists(os.path.join(args.output,'unlabel_prepared_annos/Industry/annotations/full')):
                os.makedirs(os.path.join(args.output,'unlabel_prepared_annos/Industry/annotations/full'))
        if not os.path.exists(os.path.join(args.output,'unlabel_images/full')):
                os.makedirs(os.path.join(args.output,'unlabel_images/full'))

        if not os.path.exists(os.path.join(args.output,'images/full')):
                os.makedirs(os.path.join(args.output,'images/full'))
                with open(os.path.join(args.output,'prepared_annos/Industry/train_list.txt'),'w') as f:
                    for i in os.listdir(os.path.join(args.input,'train2017')):
                        shutil.copyfile(os.path.join(args.input,'train2017',i), os.path.join(args.output,'images/full',i))
                        f.write(i+'\n')

        if not os.path.exists(os.path.join(args.output,'valid_images/full')):
                os.makedirs(os.path.join(args.output,'valid_images/full'))
                with open(os.path.join(args.output,'prepared_annos/Industry/valid_list.txt'),'w') as f:
                    for i in os.listdir(os.path.join(args.input,'val2017')):
                        shutil.copyfile(os.path.join(args.input,'val2017',i), os.path.join(args.output,'valid_images/full',i))
                        f.write(i + '\n')
        # for  annotations
        print("Converting original coco annotations to DSL-style annotations")
        data_train = json.load(open(os.path.join(args.input, 'annotations', 'instances_train2017.json')))
        print(data_train.keys())
        data_val = json.load(open(os.path.join(args.input, 'annotations', 'instances_val2017.json')))
        print(data_val.keys())
        ori_catid={}
        with open(os.path.join(args.output,'mmdet_category_info.json'),'w') as f:
                cat_info = {}
                cat_info['cat2id']={}
                cat_info['id2cat']={}
                for i in range(len(data_val['categories'])):
                        if data_val['categories'][i]['name'] not in cat_info['cat2id'].keys():
                                cat_info['cat2id'][data_val['categories'][i]['name']] = i
                                cat_info['id2cat'][str(i)] = data_val['categories'][i]['name']
                        ori_catid[str(data_val['categories'][i]['id'])]=data_val['categories'][i]['name']
                cat_info['cat2id']['背景']=len(data_val['categories'])
                cat_info['id2cat'][str(len(data_val['categories']))]='背景'
                json.dump(cat_info, f, indent=4,ensure_ascii=False)
        # init val list
        with open(os.path.join(args.output,'prepared_annos/Industry/valid_list.txt'),'r') as f:
                for line in f.readlines():
                        name = line.strip()
                        with open(os.path.join(args.output,'prepared_annos/Industry/annotations/full/',name+'.json'),'w') as f1:
                                res={}
                                res["imageName"]='full/'+name
                                res["targetNum"]=0
                                res["rects"]=[]
                                res["tags"]=[]
                                res["masks"]=[]
                                json.dump(res, f1, indent=4, ensure_ascii=False)
        # write data to init file for val list
        for i in range(len(data_val['annotations'])):
                if i%1000==0:
                        print(i)
                data=data_val['annotations'][i]
                with open(os.path.join(args.output,'prepared_annos/Industry/annotations/full/', str(data['image_id']).rjust(12,'0')+'.jpg.json'),'r') as f:
                        res=json.load(f)
                        if res["imageName"] != "full/"+str(data['image_id']).rjust(12,'0')+'.jpg':
                                print("full/"+str(data['image_id']).rjust(12,'0')+'.jpg')
                        res["targetNum"] +=1
                        res["rects"].append([data['bbox'][0],data['bbox'][1],data['bbox'][0]+data['bbox'][2], data['bbox'][1]+data['bbox'][3]])
                        res["tags"].append(ori_catid[str(data['category_id'])])
                        res["masks"].append([])
                with open(os.path.join(args.output,'prepared_annos/Industry/annotations/full/',str(data['image_id']).rjust(12,'0')+'.jpg.json'),'w') as f:
                        json.dump(res, f, indent=4, ensure_ascii=False)
        # init train list
        with open(os.path.join(args.output,'prepared_annos/Industry/train_list.txt'),'r') as f:
                for line in f.readlines():
                        name = line.strip()
                        with open(os.path.join(args.output,'prepared_annos/Industry/annotations/full/',name+'.json'),'w') as f1:
                                res={}
                                res["imageName"]='full/'+name
                                res["targetNum"]=0
                                res["rects"]=[]
                                res["tags"]=[]
                                res["masks"]=[]
                                json.dump(res, f1, indent=4, ensure_ascii=False)
        # write data to init file for train list
        for i in range(len(data_train['annotations'])):
                if i%1000==0:
                        print(i)
                data=data_train['annotations'][i]
                with open(os.path.join(args.output,'prepared_annos/Industry/annotations/full/', str(data['image_id']).rjust(12,'0')+'.jpg.json'),'r') as f:
                        res=json.load(f)
                        if res["imageName"] != "full/"+str(data['image_id']).rjust(12,'0')+'.jpg':
                                print("full/"+str(data['image_id']).rjust(12,'0')+'.jpg')
                        res["targetNum"] +=1
                        res["rects"].append([data['bbox'][0],data['bbox'][1],data['bbox'][0]+data['bbox'][2], data['bbox'][1]+data['bbox'][3]])
                        res["tags"].append(ori_catid[str(data['category_id'])])
                        res["masks"].append([])
                with open(os.path.join(args.output,'prepared_annos/Industry/annotations/full/',str(data['image_id']).rjust(12,'0')+'.jpg.json'),'w') as f:
                        json.dump(res, f, indent=4, ensure_ascii=False)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="setting input path and output path")
    parser.add_argument('--input',type=str,default=None,help="coco dir path: /gruntdata2/tcguo/coco")
    parser.add_argument('--output',type=str,default=None,help="DSL-style semicoco data dir path: /gruntdata1/bhchen/factory/data/semicoco")
    args= parser.parse_args()
    convert(args)
