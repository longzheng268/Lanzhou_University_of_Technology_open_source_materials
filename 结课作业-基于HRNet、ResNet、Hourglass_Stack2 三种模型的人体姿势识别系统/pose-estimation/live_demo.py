import torch
import cv2

from utils import draw_joints, heatmaps_to_coords
from models.hourglass import hg_stack2
from models.pose_res_net import PoseResNet
from models.hr_net import hr_w32


use_model = 'HRNet' # 可选：Hourglass_Stack2, ResNet, HRNet
ckpt = r'weights\HRNet_epoch20_loss0.000459.pth' # 模型文件

device = torch.device("cuda")



#
# 检查PyTorch是否看到GPU
print("Num GPUs Available: ", torch.cuda.device_count())

#
if use_model == 'Hourglass_Stack2':
    model = hg_stack2().to(device)
elif use_model == 'ResNet':
    model = PoseResNet().to(device)
elif use_model == 'HRNet':
    model = hr_w32().to(device)
else:
    raise NotImplementedError
model.load_state_dict(torch.load(ckpt)['model'])
model.eval()

vc = cv2.VideoCapture(0)
with torch.no_grad():
    while True:
        ret, img = vc.read()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # (h,w,c)
        img_rgb = img_rgb[:,80:-80,:]
        img_rgb = cv2.resize(img_rgb, (256,256))
        img = torch.from_numpy(img_rgb).permute(2,0,1).unsqueeze(0).float().to(device) / 255.

        if use_model in ['ResNet', 'HRNet']:
            heatmaps_pred = model(img)

        elif use_model in ['Hourglass_Stack2']:
            heatmaps_pred = model(img)[-1]

        # (1,c,h,w)
        heatmaps_pred = heatmaps_pred.double()

        heatmaps_pred_np = heatmaps_pred.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        coord_joints = heatmaps_to_coords(heatmaps_pred_np, resolu_out=[256, 256], prob_threshold=0.1)
        img_rgb = draw_joints(img_rgb, coord_joints)

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) # (h,w,c)
        img_bgr = cv2.resize(img_bgr, (600,600))
        cv2.imshow('res', img_bgr)
        c = cv2.waitKey(1)
        if c == 27:
            break

vc.release()
cv2.destroyAllWindows()

