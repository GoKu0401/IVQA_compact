# -*- coding: utf-8 -*- 
# @Time     : 2022/5/18 19:03 
# @Author   : Zhenfeng Sun
# @Email    : sunzhenfeng@hisilicon.com
# @FileName : inference_compact.py
# @Software : Pycharm

"""
这是一个精简版的 IQA inference 代码。

python inference_compact.py --model ./checkpoints/NDG.model --folder ./frames/ --imageformat png --data_type image_folders
"""

import numpy as np
import imageio

from keras.models import load_model
from keras.applications import densenet
import glob
import os
import pandas as pd
import argparse

import tensorflow.compat.v1 as tf

np.random.seed(7)

tf.Session()
tf.set_random_seed(9)


def image_infer(model, folder, imageformat):
    model_ndg = load_model(model)

    path = [x[0] for x in os.walk(folder)]
    del path[0]
    for folder_path in path:
        image_list = []
        image_name = []

        for img_path in glob.glob(folder_path + '/*.' + imageformat):
            im = np.array(imageio.imread(img_path))
            image_list.append(im)
            image_name.append(img_path.split('\\')[-1])    # 注意路径，windows '\\', 也可能 '/'

        mos = np.empty(len(image_list))
        count = 0
        for ims in image_list:

            patches = np.zeros((ims.shape[0] // 299, ims.shape[1] // 299, 299, 299, 3))

            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    patches[i, j] = ims[i * 299:(i + 1) * 299, j * 299:(j + 1) * 299]

            patches = densenet.preprocess_input(patches.reshape((-1, 299, 299, 3)))
            preds = model_ndg.predict(patches)
            avg_pred = np.mean(preds)
            mos[count] = avg_pred
            del avg_pred, preds, patches
            count = count + 1
        print("MOS prediction based on average pooling:", np.mean(mos[:]))
        name = folder_path.split('/')

        df = pd.DataFrame(mos, index=image_name, columns=['MOS'])
        df.to_csv(f'./results/{name[-1]}_predicted_mos.csv', index=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-mp', '--model', action='store', dest='model',
                        default=r'./checkpoints/NDG.model',
                        help='Specify model together with the path, e.g. ./checkpoints/NDG.model')

    parser.add_argument('-vp', '--videopath', action='store', dest='videopath', default=r'./videos/',
                        help='Specify the path of video that is going to be evaluated')

    parser.add_argument('-fr', '--videoname', action='store', dest='videoname', default='sample1.mp4',
                        help='Specify the name of the video e.g. sample.mp4')

    parser.add_argument('-fl', '--folder', action='store', dest='folder', default=r'./frames/',
                        help='Specify the path for the folder that contains folders of frames')

    parser.add_argument('-imf', '--imageformat', action='store', dest='imageformat', default='png',
                        help='Specify the format of extracted frames, e.g. png, jpg, bmp')

    parser.add_argument('-t', '--data_type', action='store', dest='data_type', default='image_folders',
                        help='Specify the data type, video or image_folders')

    values = parser.parse_args()

    if values.data_type == 'image_folders':
        image_infer(values.model, values.folder, values.imageformat)

    else:
        print("No such option")
