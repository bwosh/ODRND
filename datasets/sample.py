import cv2
import numpy as np

from scipy.ndimage.filters import gaussian_filter

from objects.bbox import *
from datasets.utils import *

class DatasetSample:
    def __init__(self, img_path: str, bbox_list: BBoxList, original_img_size=None):
        self.img_path = img_path
        self.bboxes = bbox_list
        self.original_img_size = original_img_size
        self.img_size = None
        
    def get_original_img_size(self):
        if self.original_img_size is not None:
            return self.original_img_size

        img = cv2.imread(self.img_path)
        self.original_img_size = img.shape
        return self.original_img_size

    def get_image_as_rgb_array(self, target_size=None):
        img = cv2.imread(self.img_path)
        self.original_img_size = img.shape

        # TODO convert if grayscale

        if target_size is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        self.img_size = img.shape

        return img[:,:,::-1].copy()

    def get_centers_heatmap(self, class_ids:list, target_size=None, as_image=False):
        class_maps = []

        img = self.get_image_as_rgb_array(target_size)

        oh, ow, _ = self.original_img_size
        h, w, _ = img.shape

        temp = np.zeros((h,w), dtype=float)
        for class_id in class_ids:
            heatmap = np.zeros((h,w), dtype=float)

            for bbox in self.bboxes:
                if bbox.class_id == class_id:
                    x1 = int(bbox.x1 * w / ow)
                    y1 = int(bbox.y1 * h / oh)
                    x2 = int(bbox.x2 * w / ow)
                    y2 = int(bbox.y2 * h / oh)

                    width = x2 - x1
                    height = y2 - y1

                    cx = x1 + width//2
                    cy = y1 + height//2

                    radius = gaussian_radius((width, height))

                    temp[:,:] = 0
                    temp[cy,cx] = 1
                    temp = gaussian_filter(temp, radius)

                    temp = temp/np.max(temp)
                    heatmap = np.maximum(heatmap, temp)
            class_maps.append(heatmap)

        map = np.stack(class_maps).transpose(1,2,0)
        if as_image:
            return (map * 255).astype('uint8')
        else:
            return map

    def get_height_width_maps(self, target_size=None):
        img = self.get_image_as_rgb_array(target_size)

        oh, ow, _ = self.original_img_size
        h, w, _ = img.shape

        wh = np.zeros((h,w,2), dtype=float)
        mask = np.zeros((h,w), dtype=float)
        for bbox in self.bboxes:
                x1 = int(bbox.x1 * w / ow)
                y1 = int(bbox.y1 * h / oh)
                x2 = int(bbox.x2 * w / ow)
                y2 = int(bbox.y2 * h / oh)

                width = x2 - x1
                height = y2 - y1

                cx = x1 + width//2
                cy = y1 + height//2

                wh[cy,cx,0] = w
                wh[cy,cx,0] = h
                mask[cy,cx] = 1

        return wh, mask

    def draw_bboxes(self, target_size=None):
        img = self.get_image_as_rgb_array(target_size)

        oh, ow, _ = self.original_img_size
        h, w, _ = img.shape

        for bbox in self.bboxes:
            x1 = int(bbox.x1 * w / ow)
            y1 = int(bbox.y1 * h / oh)
            x2 = int(bbox.x2 * w / ow)
            y2 = int(bbox.y2 * h / oh)
            class_id = bbox.class_id
            label = bbox.class_name
            score = bbox.score

            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),1)
            cv2.putText(img, f"{label}", (x1+2,y1+10), cv2.FONT_HERSHEY_DUPLEX,0.4, (255,255,255), lineType=cv2.LINE_AA )
            cv2.putText(img, f"{score:.2f}", (x1+2,y1+20), cv2.FONT_HERSHEY_SIMPLEX,0.3, (200,255,0), lineType=cv2.LINE_AA )

        return img