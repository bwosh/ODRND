import cv2
import numpy as np

from datasets.sample import DatasetSample
from datasets.utils import ensure_dir
from objects.bbox import *


def test_bbox_hm():
    bbox1 = BBox(0,100,140,200,     1, "vehicle", score=0.95)
    bbox2 = BBox(280,10,350,240,    0, "person",  score=0.94)
    bbox3 = BBox(140,200,200,340,   2, "animal",  score=0.93)
    bbox4 = BBox(220,80,420,370,    2, "animal",  score=0.92)
    bbox5 = BBox(430,120,450,180,   0, "person",  score=0.91)

    bbox_list = BBoxList()
    bbox_list.append(bbox1)
    bbox_list.append(bbox2)
    bbox_list.append(bbox3)
    bbox_list.append(bbox4)
    bbox_list.append(bbox5)

    sample = DatasetSample("./assets/sample.jpg", bbox_list)
    bboxes_img = sample.draw_bboxes()
    cv2.imwrite("./assets/sample_bbox.jpg", bboxes_img[:,:,::-1])
    bboxes_img = sample.draw_bboxes(target_size=(224,224))
    cv2.imwrite("./assets/sample_bbox_224.jpg", bboxes_img[:,:,::-1])

    heatmap_img = sample.get_centers_heatmap([0,1,2],as_image=True)
    cv2.imwrite("./assets/sample_hm.jpg", heatmap_img[:,:,::-1])

def test_preds(model, dataset):
    for i in range(1,10,1):

        # prepare inputs
        item = dataset[i]
        img = item.get_image_as_rgb_array(target_size=(256,256))

        # TODO adjust test preds
        return

        _, mask = item.get_height_width_maps(target_size=(32,32))
        gt_hm = item.get_centers_heatmap([0,1,2], target_size=(32,32), as_image=True)

        # Preprocess inputs
        mask[:,:,:]=0
        mask = np.expand_dims(mask, axis=0)
        img = np.expand_dims(img/255, axis=0)

        # Model interaction
        pred = model.model.predict((img, mask))

        # post-process
        hm = pred[0]
        wh = pred[1]
        pred_hm = hm[0]
        max_hm = np.max(pred_hm)
        #if max_hm!=0:
        #    pred_hm = pred_hm/max_hm
        pred_hm = (np.clip(pred_hm,0,1)*255).astype('uint8')
        print(pred_hm.shape, np.min(pred_hm), np.mean(pred_hm), np.max(pred_hm))

        # save data as images
        path_base = f"./cache/PREDS/{i}"
        ensure_dir(path_base+"_orig.png")
        cv2.imwrite(path_base+"_orig.png", img[:,:,::-1])
        cv2.imwrite(path_base+"_hm_gt.png", gt_hm)
        cv2.imwrite(path_base+"_hm_pred.png", pred_hm)
        