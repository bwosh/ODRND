import cv2
from datasets.sample import DatasetSample
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