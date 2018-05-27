from tf_models.base.data_set import BaseDataSet
import numpy as np
import tensorflow as tf
import cv2
import os

class DataSet(BaseDataSet):
    def __init__(self, options):
        super(DataSet, self).__init__(options)
        cur_dir = os.path.split(os.path.realpath(__file__))[0]
        self.data_dir = cur_dir + "/data"
        self.name_list = []
        self.get_name_list()
        self.val_name_list = self.name_list[:200]
        self.train_name_list = self.name_list[200:]

    def get_name_list(self):
        for i in range(3, 10):
            dir = os.path.join(self.data_dir, 'exp_%02d' % i)
            this_name = os.listdir(dir)
            this_name = [os.path.join(dir, name) for name in this_name]
            self.name_list = self.name_list + this_name
        self.name_list_raw = self.name_list
        self.name_list = filter(lambda name: 'res' in name, self.name_list)
        self.name_list = list(self.name_list)

        def _name_checker(name):
            posi = name.index('_res')
            img_name = name[:posi] + '.png'
            if img_name in self.name_list_raw:
                return True
            else:
                return False

        self.name_list = list(filter(_name_checker, self.name_list))

    def get_data_batch(self, batch_name):
        batch = {}
        for idx, name in enumerate(batch_name):
            posi = name.index('_res')
            img_name = name[:posi] + '.png'
            x, y = name[name.index('_h_') + 3: name.index('_h_') + 6], name[name.index('_w_') + 3: name.index('_w_') + 6]
            x, y = int(x), int(y)
            img = cv2.imread(img_name)
            # img = img[320: -320, :, :]
            # 将中间的白点去除。
            mask1 = (img[:, :, 0] == 245)
            mask2 = (img[:, :, 1] == 245)
            mask3 = (img[:, :, 2] == 245)
            mask = mask1 * mask2 * mask3
            img[mask] = img[x + 10, y + 14, :]
            x_a = np.random.randint(-50, 50)
            y_a = np.random.randint(-50, 50)

            # 截取目标点上下左右320*320的区域作为训练特征，同时调整目标点的坐标。
            x1 = x - 160 + x_a
            x2 = x + 160 + x_a
            y1 = y - 160 + y_a
            y2 = y + 160 + y_a
            x = 160 - x_a
            y = 160 - y_a
            if y1 < 0:
                y = 160 - y_a + y1
                y1 = 0
                y2 = 320
            if y2 > img.shape[1]:
                y = 160 - y_a + y2 - img.shape[1]
                y2 = img.shape[1]
                y1 = y2 - 320
            img = img[x1: x2, y1: y2, :]
            label = np.array([x, y], dtype=np.float32)

            if idx == 0:
                batch['img'] = img[np.newaxis, :, :, :]
                batch['label'] = label.reshape([1, label.shape[0]])
            else:
                img_tmp = img[np.newaxis, :, :, :]
                label_tmp = label.reshape((1, label.shape[0]))
                batch['img'] = np.concatenate((batch['img'], img_tmp), axis=0)
                batch['label'] = np.concatenate((batch['label'], label_tmp), axis=0)
        return batch['img'], batch['label']

    def next_batch(self, batch_size=8):
        batch_name = np.random.choice(self.train_name_list, batch_size)
        x, y = self.get_data_batch(batch_name)
        yield x, y

    def test_set(self):
        x, y = self.get_data_batch(self.val_name_list)
        return x, y
