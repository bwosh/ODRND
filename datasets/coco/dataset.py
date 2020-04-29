import json
import os

from collections import defaultdict
from tqdm import tqdm

from datasets.base import Dataset
from datasets.sample import DatasetSample
from datasets.utils import download, ensure_dir

class CocoDataset(Dataset):
    def __init__(self, coco_subset_name:str, annot_path:str, supercategories: list):
        super().__init__(f"COCO_{coco_subset_name}")
        self.supercategories = supercategories
        self.coco_subset_name = coco_subset_name

        # Load file
        print("Loading COCO subset:", coco_subset_name,"for",supercategories,'...')
        with open(annot_path, "r") as file:
            annotations_data =json.load(file)

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

        self.download_images(images_annots)
        self.images_bboxes = images_bboxes

    def download_images(self, images_annots):
        ann_dict = {}
        for i in images_annots:
            image_id = i['id']
            url = i['coco_url']
            ann_dict[image_id] = url

        for image_id in tqdm(ann_dict):
            source_url = ann_dict[image_id]
            _, file_extension = os.path.splitext(os.path.basename(source_url))
            target_path = f"./cache/{self.coco_subset_name}/{image_id}{file_extension}"
            if not os.path.isfile(target_path):
                ensure_dir(target_path)
                download(source_url, target_path)

    def __len__(self):
        raise Exception("Not implmented")

    def __getitem__(self, index) -> DatasetSample:
        raise Exception("Not implmented")   