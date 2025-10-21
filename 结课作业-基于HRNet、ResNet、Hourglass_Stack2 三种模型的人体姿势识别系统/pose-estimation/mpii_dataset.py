import numpy as np
from torch.utils.data import Dataset
import skimage
import json

from utils import load_image, crop, change_resolu, generate_heatmaps

import os

class MPIIDataset(Dataset):
    def __init__(self, is_train=True, use_scale=True, use_flip=True, use_rand_color=True):
        self.is_trian = is_train
        self.use_scale = use_scale
        self.use_flip = use_flip
        self.use_rand_color = use_rand_color
        
        # Get json annotation
        with open('data/mpii_human_pose_v1/mpii_annotations.json') as anno_file:
            self.anno = json.load(anno_file)  # len 25204

        self.train_list, self.valid_list = [], []  # 22246, 2958
        for idex, ele_anno in enumerate(self.anno):
            if ele_anno['isValidation'] == True:
                self.valid_list.append(idex)
            else:
                self.train_list.append(idex)

    def __getitem__(self, idx):
        if self.is_trian:
            ele_anno = self.anno[self.train_list[idx]]
        else:
            ele_anno = self.anno[self.valid_list[idx]]
        res_img = [256, 256]
        res_heatmap = [64, 64]
        path_img_folder = 'data/mpii_human_pose_v1/images'
        path_img = os.path.join(path_img_folder, ele_anno['img_paths'])
        img_origin = load_image(path_img)

        img_crop, ary_pts_crop, c_crop = crop(img_origin, ele_anno, use_randscale=self.use_scale,
                                                        use_randflipLR=self.use_flip, use_randcolor=self.use_rand_color)

        img_out, pts_out, _ = change_resolu(img_crop, ary_pts_crop, c_crop, res_heatmap)

        train_img = skimage.transform.resize(img_crop, tuple(res_img))
        train_heatmap = generate_heatmaps(img_out, pts_out, sigma_valu=2)
        train_pts = pts_out[:, :2].astype(np.int32)

        # (H,W,C) -> (C,H,W)
        train_img = np.transpose(train_img, (2, 0, 1))
        train_heatmap = np.transpose(train_heatmap, (2, 0, 1))

        return train_img, train_heatmap, train_pts

    def __len__(self):
        if self.is_trian:
            return len(self.train_list)
        else:
            return len(self.valid_list)

if __name__ == '__main__':
    dataset = MPIIDataset()
    img, heatmap, pts = dataset[0]
    print(img.shape) # (3,256,256)
    print(heatmap.shape) # (16,64,64)
    print(pts.shape) # (16,2)
    