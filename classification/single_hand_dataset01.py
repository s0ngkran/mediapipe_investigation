from torch.utils.data import Dataset
import json
import torch

class MyDataset(Dataset):
    def __init__(self, json_path, test_mode=False):
        with open(json_path, 'r') as f:
            dat = json.load(f)
        print('-------------n',len(dat))
        if test_mode:
            dat = dat[:50]
        
        self.img_path = []
        self.ground_truth = []
        self.handedness = []
        self.hand_landmarks = []

        for _dat in dat:
            _img_path = _dat['img_path']
            _ground_truth = _dat['ground_truth']
            _handedness = _dat['handedness']
            _hand_landmark = _dat['hand_landmarks']

            my_landmark = []
            if _hand_landmark == '':
                continue
            assert len(_hand_landmark[0]) == 21

            # get relative to index0
            found_index0 = False
            for i, point in enumerate(_hand_landmark[0]):
                if found_index0:
                    x_rel = x_index0 - point['X']
                    y_rel = y_index0 - point['Y']
                    z_rel = z_index0 - point['Z']
                    my_landmark.append(x_rel)
                    my_landmark.append(y_rel)
                    my_landmark.append(z_rel)

                if i == 0:
                    x_index0 = point['X']
                    y_index0 = point['Y']
                    z_index0 = point['Z']
                    found_index0 = True
            assert len(my_landmark) == 60
                

            gt_dict = {
                # number
                'Num01': 0,
                'Num02': 1,
                'Num03': 2,
                'Num04': 3,
                'Num05': 4,

                # addition
                'Add01': 5,
                'Add02': 6,
                'Add03': 7,
                'Add04': 8,
                'Add05': 9,

                # static pose
                'A': 10,
                'B': 11,
                'D': 12,
                'F': 13,
                'H': 14,
                'K': 15,
                'L': 16,
                'M': 17,
                'N': 18,
                'P': 19,
                'R': 20,
                'S': 21,
                'T': 22,
                'W': 23,
                'Y': 24,

                # static pose for vowel
                'Vow01': 25,
                'Vow02': 26,
                'Vow03': 27,
                'Vow04': 28,
                'Vow05': 29,
            }
            thai_to_poses = {
                # static mode
                '???':['K'],
                '???':['T'],
                '???':['P'],
                '???':['H'],
                '???':['B'],
                '???':['R'],
                '???':['W'],
                '???':['D'],
                '???':['F'],
                '???':['L'],
                '???':['Y'],
                '???':['M'],
                '???':['N'],
                '???':['A'],
                '??????':['Vol01'],
                '??????':['Vol03'],
                '??????':['Vol04'],
                '??????':['Vol05'],

                # dynamic mode
                '???':['K','Num01'],
                '???':['K','Num02'],
                '???':['K','Num03'],

                '???':['T','Num01'],
                '???':['T','Num02'],
                '???':['T','Num03'],
                '???':['T','Num04'],
                '???':['T','Num05'],

                '???':['S','Num01'],
                '???':['S','Num02'],

                '???':['S','Add03'],

                '???':['P','Num01'],
                '???':['P','Num02'],
                '???':['P','Num03'],

                '???':['H','Num01'],
                '???':['D','Num01'],
                '???':['F','Num01'],
                '???':['L','Num01'],
                '???':['Add04','Add05'], # ?????????????????????????????? ????????????????????? ???????????? ?????????????????????
                '???':['Y','Num01'],

                '???':['N','Num01'],
                '???':['N','Add01'],

                '???':['T','H'],
                '???':['T','H','Num01'],

                '???':['Add02','H'],
                '???':['Add02','H', 'Num01'],
                '???':['Add02','H', 'Num02'],

                '??????':['Vow01','Num01'],
                '??????':['Vow01','Vow02'],
                '??????':['Vow01','Num02'],
            }
            self.img_path.append(_img_path)
            self.ground_truth.append(gt_dict[_ground_truth])
            self.handedness.append(_handedness)
            self.hand_landmarks.append(torch.FloatTensor(my_landmark))
        
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        ans = {
            'img_path': self.img_path[idx],
            'ground_truth': self.ground_truth[idx],
            'handedness': self.handedness[idx],
            'hand_landmarks': self.hand_landmarks[idx],
        }
        return ans
   