from torch.utils.data import Dataset
import copy
from PIL import Image
import json
import torch
from torchvision import transforms
import cv2

class MyDataset(Dataset):
    def __init__(self, json_path, test_mode=False):
        with open(json_path, 'r') as f:
            dat = json.load(f)
        print('-------------n_data=',len(dat))
        if test_mode:
            dat = dat[:50]
        
        self.img_path = []
        # try to pre-load images
        if ('testing_set' in json_path):
            aug = True
            # aug
            if aug:
                self.transform = transforms.Compose([
                    # transforms.Rotation(10,fill=(0,0,0)),

                    transforms.ColorJitter(
                        brightness=0.03,
                        contrast=(0, 0.15),
                        saturation=(0, 0.09),
                        hue=(0, 0.04),
                        ),
                    transforms.Resize((64, 64)), 
                    transforms.ToTensor()
                ])
            # no aug
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((64, 64)), 
                    transforms.ToTensor()
                ])
        else:
            self.transform = transforms.Compose([
                # transforms.RandomRotation((0,15),fill=(0,0,0)),
                # transforms.RandomCrop(54, fill=(0,0,0)),

                transforms.ColorJitter(
                    brightness=(0, 0.1),
                    contrast=(0, 0.1),
                    saturation=(0, 0.1),
                    hue=(0, 0.1),
                    ),
                transforms.Resize((64, 64)), 
                transforms.ToTensor()
            ])
        self.ground_truth = []

        def get_gt_dict():
            labels = ['A','B','C','D','E','F','G','H','I','J','K']
            nums = [i for i in range(11)]
            gt_dict = {}
            for label, num in zip(labels, nums):
                gt_dict.update({
                    str(label):int(num),
                })
            return gt_dict
        gt_dict = get_gt_dict()

        zero_gt = [0 for i in range(11)]
        def gen_one_hot(n):
            gt = copy.deepcopy(zero_gt)
            gt[n] = 1
            return gt

        folder = './poh_data/'
        for _dat in dat:
            _img_path = _dat['img_path'].replace('./',folder)
            _ground_truth = _dat['gt']

            
            self.img_path.append(_img_path)
            self.ground_truth.append(gt_dict[_ground_truth])
        
    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        image = Image.open(self.img_path[idx])
        ans = {
            'img': self.transform(image),
            'gt': self.ground_truth[idx],
        }
        return ans

if __name__ == '__main__':
    # folder -> ./poh_data/
    json_path = './poh_data/replaced/replaced_bg.poh_black_training.json'
    dataset = MyDataset(json_path)
    print('n_data=', len(dataset))

   
