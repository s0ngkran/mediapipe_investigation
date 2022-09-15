import json
import numpy as np
import math
import os
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


class Point:
    def __init__(self, data=None, x=None, y=None):
        if x == None and y == None:
            self.x = data['X']
            self.y = data['Y']
            self.z = data['Z']
        else:
            self.x = x
            self.y = y
        self.dist = None
        self.tag = None
        self.index = None

    def distance(self, p, w=None, h=None):
        dist = ((p.x  - self.x)*w)**2 + ((p.y - self.y)*h)**2
        dist = dist**0.5
        return dist



class Data:
    def __init__(self, data):
        self.dataset_name = 'poh_data_with_mph'
        self.img_path = data['img_path']
        self.ground_truth = data['ground_truth']
        self.handedness = data['handedness']
        self.hand_landmarks = data['hand_landmarks']
        self.img_name = self.img_path.split('/')[-1]
        self.gt = None
        img = cv2.imread(self.img_path)
        self.img_w, self.img_h, channel = img.shape

        # const
        self.finger_index = [0, 2, 4, 5, 8, 9, 12, 13, 16, 17]
        self.finger_label = ['A', 'B', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'J']

        # create gt map
        def create_gt_map():
            label = self.finger_label
            index = self.finger_index
            gt_map = {}
            for lab, ind in zip(label, index):
                gt_map[str(ind)] = lab
            return gt_map
        self.gt_map = create_gt_map()

        # print(self.img_path)

        def read_hands(data):
            hands = []
            for hand_ in data:
                hand = []
                for point in hand_:
                    hand.append(Point(point))
                hands.append(hand)
            if hands == []:
                print('empty hand')
            return hands
        self.hands = read_hands(self.hand_landmarks)
        self.n_hand = len(self.hands)
        self.point_k = None
        self.palm_hand = None
        self.pointing_hand = None
        for hand in self.hands:
            p0 = hand[0]
            p9 = hand[9]
            x_mean = (p0.x + p9.x)/2
            y_mean = (p0.y + p9.y)/2
            is_pointing = self.is_pointing_hand(hand)
            if is_pointing:
                self.pointing_hand = hand
            else: # this is palm hand
                self.palm_hand = hand
                self.point_k = Point(x=x_mean, y=y_mean)

        #############
        # assert self.n_hand == 1
        # print('n_hand=',self.n_hand)
        # self.hand1 = self.hands[0]
        # self.hand2 = self.hands[1]

    def pred_nearest(self):
        try:
            assert self.gt is not None
            assert self.palm_hand is not None
            assert self.pointing_hand is not None
        except:
            return False

        # dist = target.distance(pointing, w=self.img_w, h=self.img_h)

        pointing = self.pointing_hand[8] # index_finger_tip
        hand = self.palm_hand
        min_dist = 100000 # big number
        min_index = None
        min_point = None
        for index, i in enumerate(self.finger_index):
            p = hand[i]
            dist = pointing.distance(p, self.img_w, self.img_h)
            if dist < min_dist:
                min_dist = dist
                min_index = i
                min_point = p
        dist = pointing.distance(self.point_k, self.img_w, self.img_h)
        if dist < min_dist:
            pred = self.point_k
            pred_label = 'K'
        else:
            pred = min_point
            pred_label = self.gt_map[str(min_index)]
            
        plt.plot(pred.x*self.img_w, pred.y*self.img_h, 'og')
        if pred_label == self.gt:
            return True
        else:
            return False
            
        # print('dist', dist)


    def plot(self):
        img = cv2.imread(self.img_path)
        w, h, channel = img.shape

        def plot_line(p1, p2, w=w, h=h, color='-y'):
            x1, y1 = p1.x * w, p1.y*h
            x2, y2 = p2.x * w, p2.y*h
            plt.plot((x1, x2), (y1, y2), color)
        img = cv2.flip(img, 1)
        plt.imshow(img[:, :, (2, 1, 0)])
        plt.title('gt='+str(self.gt))
        for hand_index, hand in enumerate(self.hands):
            is_pointing_hand = self.is_pointing_hand(hand)
            for i, p in enumerate(hand):
                if i in self.finger_index:
                    if i in [8, 5] and is_pointing_hand:  # index finger
                        color = 'or'
                    else:
                        color = 'ob'
                    plt.plot(p.x * w, p.y*h, color)
            plot_line(hand[0], hand[2])
            plot_line(hand[0], hand[9])
            plot_line(hand[2], hand[4])
            plot_line(hand[5], hand[8])
            if is_pointing_hand:
                plot_line(hand[5], hand[8], color='-r')
            plot_line(hand[9], hand[12])
            plot_line(hand[13], hand[16])
            # plt.text(p.x * w, p.y*h, str(i))
        k = self.point_k
        plt.plot(k.x*w, k.y*h, 'ob')
        self.pred_nearest()
        plt.show()

    def is_pointing_hand(self, hand):
        def vector_dir(p_start, p_end):
            distance = [p_end.x - p_start.x, p_end.y - p_start.y]
            norm = (distance[0] ** 2 + distance[1] ** 2)**0.5
            direction = [distance[0] / norm, distance[1] / norm]
            # example
            # d = vector_dir(Point(x=0,y=0), Point(x=10,y=0))
            return direction

        def angle_between(v0, v1):
            angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
            angle = np.degrees(angle)
            angle = abs(angle)
            return angle

        index_finger = vector_dir(hand[5], hand[8])  # index_finger
        middle_finger = vector_dir(hand[9], hand[12])  # middle_finger
        ring_finger = vector_dir(hand[13], hand[16])
        pink_finger = vector_dir(hand[17], hand[20])

        a = angle_between(index_finger, middle_finger)
        b = angle_between(index_finger, ring_finger)
        c = angle_between(index_finger, pink_finger)
        mean = sum([a, b, c])/3
        if mean > 90:
            return True
        return False

    def add_gt(self, all_gt):
        for gt in all_gt:
            if gt.img_name == self.img_name:
                self.gt = gt.gt
                return True


def read_data(path, all_gt) -> list:
    with open(path, 'r') as f:
        data = json.load(f)
    print('loaded', path, '->', len(data))
    ans = []
    for dat in data:
        _ = Data(dat)
        _.add_gt(all_gt)
        ans.append(_)
    return ans


class GroundTruth:
    def __init__(self, data):
        # self.img_path = data['img_path']
        self.gt = data['gt']
        self.img_path = os.path.join(
            '.', 'poh_data', data['img_path'].replace('./processed/', ''))
        self.img_name = self.img_path.split('/')[-1]


def read_gt(path) -> list:
    with open(path, 'r') as f:
        data = json.load(f)
    print('loaded', path, '->', len(data))
    ans = []
    for dat in data:
        ans.append(GroundTruth(dat))

    # o = {}
    # for i in ans:
    #     gt = i.gt
    #     o[str(gt)] = 'x'
    # print('---')
    # for k, v in o.items():
    #     print(k)
    # print('--')
    # print(ans[0].img_path)
    return ans

def get_path(num):
    assert num in [0,1]
    if num ==0:
        gt_path = './poh_data/processed/poh_black_testing.json'
        path = './ans_processed.json'
    elif num == 1:
        gt_path = './poh_data/replaced/replaced_bg.poh_black_testing.json'
        path = './ans_replaced.json'
    return gt_path, path

class Ans:
    def __init__(self):
        self.correct = {}
        self.fail = {}
        self.cnt = 0
    def add(self, key, is_correct):
        self.cnt += 1
        if is_correct:
            if key in self.correct:
                self.correct[str(key)] += 1
            else:
                self.correct[str(key)] = 1
        else:
            if key in self.fail:
                self.fail[str(key)] += 1
            else:
                self.fail[str(key)] = 1

    def display(self):
        for k, v in self.correct.items():
            percent =v/self.cnt*100
            correct = v
            sum = self.fail[k]+v
            print('{} {}/{} ---{:.2f}%'.format(k, correct, sum, percent))




def main():
    path_num = 1 # 0 or 1
    ans = Ans()
    gt_path, path = get_path(path_num)
    data = read_data(path, read_gt(gt_path))
    only_testing_set = []
    for d in data:
        if d.gt != None:
            only_testing_set.append(d)
    print('cnt', len(only_testing_set))
    data = only_testing_set
    # data[0].is_pointing_hand()
    correct = 0
    for dat in data:
        a = dat.pred_nearest()
        if a == True:
            correct += 1
        ans.add(dat.gt, a)
    print('correct {}/{} ---{:.2f}%'.format(correct,len(data), correct/len(data)*100))
    ans.display()


    # print(data[0].hands[0][0].x)
if __name__ == '__main__':
    main()
