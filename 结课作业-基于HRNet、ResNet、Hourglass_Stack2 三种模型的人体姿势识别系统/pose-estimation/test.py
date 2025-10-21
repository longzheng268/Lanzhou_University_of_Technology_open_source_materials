import torch

import matplotlib.pyplot as plt
import skimage

from utils import draw_joints, load_image, show_heatmaps, heatmaps_to_coords
from models.hourglass import hg_stack2
from models.pose_res_net import PoseResNet
from models.hr_net import hr_w32


use_model = 'Hourglass_Stack2' # 可选：Hourglass_Stack2, ResNet, HRNet
ckpt = 'weights/Hourglass_Stack2_epoch1_loss0.002647276851348579.pth' # 模型文件
path_testimg = 'data/test_imgs/000402528.jpg' # 测试图片

device = torch.device('cuda:0')

if use_model == 'Hourglass_Stack2':
    model = hg_stack2().to(device)
elif use_model == 'ResNet':
    model = PoseResNet().to(device)
elif use_model == 'HRNet':
    model = hr_w32().to(device)
else:
    raise NotImplementedError
model.load_state_dict(torch.load(ckpt))
model.eval()

img_np = load_image(path_testimg)
img_np = skimage.transform.resize(img_np, [256,256])
img_np_copy = img_np

img = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float().to(device)

if use_model in ['ResNet', 'HRNet']:
    heatmaps_pred = model(img)
elif use_model in ['Hourglass_Stack2']:
    heatmaps_pred = model(img)[-1]

# (1,c,h,w)
heatmaps_pred = heatmaps_pred.double()

heatmaps_pred_np = heatmaps_pred.squeeze(0).permute(1,2,0).detach().cpu().numpy()

show_heatmaps(img_np_copy, heatmaps_pred_np)

coord_joints = heatmaps_to_coords(heatmaps_pred_np, resolu_out=[256, 256], prob_threshold=0.1)
img_rgb = draw_joints(img_np_copy, coord_joints)
plt.imshow(img_rgb)
plt.show()
