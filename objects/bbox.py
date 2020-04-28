class BBox:
    def __init__(self, x1:float, y1:float, x2:float, y2:float, class_id:int, class_name:str, score:float=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.class_id = class_id
        self.class_name = class_name

        self.score = score

class BBoxList:
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def append(self, bbox: BBox):
        self.data.append(bbox)

    def __getitem__(self, index):
        return self.data[index]