import json
import matplotlib.pyplot as plt
import numpy as np

COLOR_LIST = ['.r', '.b', '.y', '.g']

class Keypoint:
    def __init__(self, data):
        self.x = data['x']
        self.y = data['y']
        self.covered = data['covered']

class BBox:
    def __init__(self, x_min, x_max, y_min, y_max, img_w=1280, img_h=720):
        num_const = 0
        percent = (x_max - x_min) * num_const
        self.x_min = x_min - percent if x_min-percent > 0 else 0
        self.x_max = x_max + percent if x_min-percent < img_w else img_w
        percent = (y_max - y_min) * num_const
        self.y_min = y_min - percent if y_min-percent > 0 else 0
        self.y_max = y_max + percent if y_min-percent < img_h else img_h
    def get_bbox(self):
        return self.x_min, self.y_min, self.x_max, self.y_max

class Data:
    def __init__(self, data, root):
        self.root = root
        self.image_path = None
        self.meta = None
        self.n_hand = None
        self.hands = []
        self.hand_bboxs = []
        assert type(data) == dict
        self.image_path = data['image_path'].replace('hand/', self.root + '/images/')
        self.meta = data['meta']
        hand_data = data['keypoints']
        self.n_hand = len(hand_data) #2
        for i, hand_data_ in enumerate(hand_data):
            hand = []
            for keypoint_data in hand_data_:
                hand.append(Keypoint(keypoint_data))
            self.hands.append(hand)
        assert self.n_hand == len(self.hands), 'n_hand=%d len(hands)=%d'%(self.n_hand, len(self.hands))


        def get_palm_bbox():
            assert type(self.hands) == list
            # find min - max
            for hand in self.hands:
                palm = []
                for i, kp in enumerate(hand):
                    if i in [0,1,3,4,5,6]:
                        palm.append(kp)
                
                palm.sort(key=lambda x: x.x)
                x_min = palm[0].x
                x_max = palm[-1].x
                palm.sort(key=lambda x: x.y)
                y_min = palm[0].y
                y_max = palm[-1].y
                bbox = BBox(x_min, x_max, y_min, y_max)
                self.hand_bboxs.append(bbox)
        get_palm_bbox()

class DatasetForMPHTesting:
    def __init__(self, folder_path):
        self.name = folder_path.split('/')[-1]
        self.root = folder_path
        self.meta_path = folder_path + '/meta.json'
        self.data = []
        with open(self.meta_path, 'r') as f:
            data = json.load(f)
        for dat in data:
            dat = Data(dat, self.root)
            self.data.append(dat)
        print('init %d data <- %s'%(len(self.data), self.meta_path))

    def plot_keypoint_on_image(self, index=0, img=None, plotxy=None, only_plot=False):
        data = self.data[index]
        def plot_rect(x_min, x_max, y_min, y_max, color):
            x1, y1 = x_min, y_min
            x2, y2 = x_min, y_max
            x3, y3 = x_max, y_max
            x4, y4 = x_max, y_min
            plt.plot((x1,x2,x3,x4,x1), (y1,y2,y3,y4,y1), color=color, linewidth=3)

        hands = data.hands
        hand_bbox = data.hand_bboxs

        if img is None:
            path = data.image_path
            img = plt.imread(path)
        
        if only_plot == False:
            plt.imshow(img)
        for color_index, hand in enumerate(hands):
            for label_index, kp in enumerate(hand):
                plt.plot(kp.x, kp.y, COLOR_LIST[color_index])
                plt.text(kp.x, kp.y, str(label_index))
        for color_index, hand in enumerate(hand_bbox):
            plot_rect(hand.x_min, hand.x_max, hand.y_min, hand.y_max, COLOR_LIST[color_index][1:])
        if plotxy is not None:
            x_min = plotxy[0]
            x_max = plotxy[1]
            y_min = plotxy[2]
            y_max = plotxy[3]
            plot_rect(x_min, x_max, y_min, y_max, 'green')
        if only_plot == False:
            plt.show()
        return hand.x_min, hand.y_min, hand.x_max, hand.y_max

def read_single_hand():
    overlap = DatasetForMPHTesting('./overlap_hand/single_hand')
    return overlap
def read_overlap0():
    overlap = DatasetForMPHTesting('./overlap_hand/overlap0')
    return overlap
def read_overlap5():
    overlap = DatasetForMPHTesting('./overlap_hand/overlap5')
    return overlap
def read_overlap20():
    overlap = DatasetForMPHTesting('./overlap_hand/overlap20')
    return overlap
def read_overlap50():
    overlap = DatasetForMPHTesting('./overlap_hand/overlap50')
    return overlap
def read_overlap80():
    overlap = DatasetForMPHTesting('./overlap_hand/overlap80')
    return overlap

def main():
    overlap0 = DatasetForMPHTesting('./overlap_hand/overlap0')
    overlap0.plot_keypoint_on_image(2)
    print('end')


if __name__ == '__main__':
    main()
