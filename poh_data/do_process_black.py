import matplotlib.pyplot as plt
import json
import cv2
import os
import time
import random
import sys
sys.path.append('..')

folder = 'raw/'

def get_thres_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th3

def get_remove_background(img, binary):
    img[binary == 0] = (0,0,0)
    return img

def save_to_processed_folder(img_name, processed_img):
    path = './processed/images/%s'%img_name
    res = cv2.imwrite(path, processed_img)
    assert res == True
    return path

# read imgs
i_img = 0
photo_folder = folder 
testing_json = []
validation_json = []
training_json = []
for folder_name_ in get_list_folder(photo_folder):
    folder_name = os.path.join(photo_folder, folder_name_)
    cnt = 0
    for img_path in get_list_img(folder_name):
        img_path = os.path.join(folder_name, img_path)
        img = cv2.imread(img_path)
        binary = get_thres_img(img)
        remove_background = get_remove_background(img, binary)
        img_name = img_path.split('/')[-1]
        path = save_to_processed_folder(img_name, remove_background)
        gt = img_path.split('/')[-2]

        # sep set
        cnt += 1
        dat = {
            'img_path': path,
            'gt': gt,
        }
        if cnt <= 36: # set first 36 imgs to testing_set
            testing_json.append(dat)
        elif cnt <= 36+30: # set 30 imgs to validation_set
            validation_json.append(dat)
        else: # remaining as training_set
            training_json.append(dat)

with open('processed/poh_black_training.json', 'w') as f:
    json.dump(training_json, f)
    print('training_set', len(training_json))
with open('processed/poh_black_validation.json', 'w') as f:
    json.dump(validation_json, f)
    print('validation_json', len(validation_json))
with open('processed/poh_black_testing.json', 'w') as f:
    json.dump(testing_json, f)
    print('testing_json', len(testing_json))