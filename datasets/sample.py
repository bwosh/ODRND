import cv2
from objects.bbox import *

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

    def get_centers_heatmap(self, class_ids:list, target_size=None):
        # TODO heatmap
        return self.get_image_as_rgb_array(target_size)

    def get_height_width_map(self, target_size=None):
        # TODO h_w map
        return self.get_image_as_rgb_array(target_size)

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