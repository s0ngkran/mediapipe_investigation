import random
import cv2
import json
import os
import matplotlib.pyplot as plt

# read random background
background_folder = './replaced/background/'
for _, __, background_paths in os.walk(background_folder):
    print('walking...', background_folder)

print('get background =>', len(background_paths), 'imgs')

def get_random_background(background_folder, background_paths):
    path = random.choices(background_paths)[0]
    path = os.path.join(background_folder, path)
    random_background = cv2.imread(path)
    return random_background

# read imgs from processed folder
processed_folder = './processed/'
files = ['poh_black_testing.json',
         'poh_black_training.json', 'poh_black_validation.json']
json_dat = []
for fname in files:
    json_path = os.path.join(processed_folder, fname)
    print('fname', fname)
    print('reading...', json_path)
    with open(json_path, 'r') as f:
        data = json.load(f)

    print('len => ', len(data))

    # get image
    for dat in data:
        img_path = dat['img_path']
        gt = dat['gt']

# get binary area
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        black_area = gray < 10
        # black_area = img == (0, 0, 0)

        background_img = get_random_background(
            background_folder, background_paths)

# replace my blackgroud
        img[black_area] = background_img[black_area]

# save to replaced folder
        new_folder = os.path.join('./replaced/', 'images')
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
            print('mkdir', new_folder)

        path = os.path.join(new_folder, img_path.split('/')[-1])
        res = cv2.imwrite(path, img)
        print('.', end='')
        json_dat.append({
            'img_path': path,
            'gt': gt,
            'meta': 'replaced-BG',
        })

# save meta json
    with open(os.path.join('./replaced/','replaced_bg.'+fname), 'w') as f:
        json.dump(json_dat, f)
