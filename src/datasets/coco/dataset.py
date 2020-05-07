import json
import os
import pickle

from collections import defaultdict
from tqdm import tqdm

from datasets.base import Dataset
from datasets.sample import DatasetSample
from datasets.utils import download, ensure_dir

from objects.bbox import BBox, BBoxList

class CocoDataset(Dataset):
    def __init__(self, coco_subset_name:str, annot_path:str, supercategories: list, opts):
        super().__init__(f"COCO_{coco_subset_name}")
        self.supercategories = supercategories
        self.coco_subset_name = coco_subset_name
        self.opts = opts

        # Load file
        print("Loading COCO subset:", coco_subset_name,"for",supercategories,'...')
        cache_path = annot_path+"_cache.pkl"
        if os.path.isfile(cache_path):
            with open(cache_path,'rb') as file:
                self.image_ids, self.images_bboxes, self.filenames = pickle.load(file)
                print(f"Loaded from cache ({len(self.image_ids)} images)...")
                return
        else:
            with open(annot_path, "r") as file:
                annotations_data = json.load(file)

        categories = annotations_data['categories']
        annotations = annotations_data['annotations']
        images_annots = annotations_data['images']

        # Get interesting supercategories
        cat_dict = defaultdict(list)

        for c in categories:
            if c['supercategory'] in supercategories:
                cat_dict[c['supercategory']].append((c['id'],c['name']))

        self.category_mapping = {}

        for c in cat_dict:
            for i in cat_dict[c]:
                self.category_mapping[i[0]]=self.supercategories.index(c)

        # Load interesting instances
        images_with_crowds = []
        images_bboxes = defaultdict(list)

        for ann in annotations:
            bbox = ann['bbox']
            cat = ann['category_id']
            crowd = ann['iscrowd']>0
            image_id = ann['image_id']

            if cat in self.category_mapping.keys():
                if crowd:
                    images_with_crowds.append(image_id)
                else:
                    images_bboxes[image_id].append((bbox, self.category_mapping[cat]))

        for image_id in images_with_crowds:
            if image_id in images_bboxes:
                del images_bboxes[image_id]
        print(f"{len(images_bboxes)} images meet criteria.")

        self.filenames = {}
        self.download_images(images_bboxes, images_annots)
        self.images_bboxes = images_bboxes
        self.image_ids = list(self.images_bboxes.keys())

        with open(cache_path,'wb') as file:
            pickle.dump((self.image_ids, self.images_bboxes, self.filenames), file)

    def download_images(self, images_bboxes, images_annots):
        ann_dict = {}
        for i in images_annots:
            image_id = i['id']
            url = i['coco_url']
            if image_id in images_bboxes:
                ann_dict[image_id] = url

        for image_id in tqdm(ann_dict):
            source_url = ann_dict[image_id]
            _, file_extension = os.path.splitext(os.path.basename(source_url))
            target_path = os.path.join(self.opts.cache_path,f"{self.coco_subset_name}/{image_id}{file_extension}")
            self.filenames[image_id] = target_path
            if not os.path.isfile(target_path):
                ensure_dir(target_path)
                download(source_url, target_path)

    def __len__(self):
        return len(self.images_bboxes)

    def __getitem__(self, index) -> DatasetSample:
        image_id = self.image_ids[index]
        bboxes = self.images_bboxes[image_id]
        filename = self.filenames[image_id]

        bbox_list = BBoxList()

        for bb in bboxes:
            (x1,y1,w,h), class_id = bb
            x2 = x1+w
            y2 = y1+h
            bbox = BBox(x1,y1,x2,y2,class_id,str(class_id))
            bbox_list.append(bbox)

        sample = DatasetSample(filename, bbox_list) # TODO IMPROVEMENT add original image size ?
        return sample