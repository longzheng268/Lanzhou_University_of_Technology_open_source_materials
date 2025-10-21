import os
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from visdom import Visdom

from models.hourglass import hg_stack2
from models.pose_res_net import PoseResNet
from models.hr_net import hr_w32
from joints_mse_loss import JointsMSELoss
from mpii_dataset import MPIIDataset
from utils import heatmaps2rgb

def calculate_accuracy(pred_heatmaps, gt_heatmaps, threshold=0.5):
    """
    计算关键点检测的准确率

    Args:
        pred_heatmaps: 预测的热图
        gt_heatmaps: 真实的热图
        threshold: 热图峰值判定阈值

    Returns:
        准确率 (0-1之间的浮点数)
    """
    pred_peaks = pred_heatmaps.max(dim=-1)[0].max(dim=-1)[0]
    gt_peaks = gt_heatmaps.max(dim=-1)[0].max(dim=-1)[0]
    correct_points = (pred_peaks > threshold) == (gt_peaks > threshold)
    accuracy = correct_points.float().mean().item()
    return accuracy

def train(config):
    # 配置训练参数
    seed = config.get('seed', 999)
    use_model = config.get('use_model', 'HRNet')
    lr = config.get('lr', 1e-3)
    batch_size = config.get('batch_size', 8)
    num_epochs = config.get('num_epochs', 20)
    ckpt = config.get('ckpt', None)
    num_workers = config.get('num_workers', 0)

    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # 设置CUDA
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'使用设备: {device}')
    print(f'使用模型: {use_model}')
    print(f'数据加载工作进程数: {num_workers}')
    print(f'训练轮次: {num_epochs}')

    # 创建权重保存目录
    os.makedirs('weights', exist_ok=True)

    # 数据集和加载器
    dataset = MPIIDataset(use_scale=True, use_flip=True, use_rand_color=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # 选择模型
    if use_model == 'Hourglass_Stack2':
        model = hg_stack2().to(device)
    elif use_model == 'ResNet':
        model = PoseResNet().to(device)
    elif use_model == 'HRNet':
        model = hr_w32().to(device)
    else:
        raise NotImplementedError(f"未知模型: {use_model}")

    # 优化器和学习率调度器
    optimizer = Adam(model.parameters(), lr=lr)
    lr_scheduler = MultiStepLR(optimizer, [10,15], .1)
    criteon = JointsMSELoss().to(device)

    # 从检查点恢复训练
    ep_start = 1
    if ckpt:
        weight_dict = torch.load(ckpt)
        model.load_state_dict(weight_dict['model'])
        optimizer.load_state_dict(weight_dict['optim'])
        lr_scheduler.load_state_dict(weight_dict['lr_scheduler'])
        ep_start = weight_dict['epoch'] + 1

    # 目标权重
    target_weight = np.array([[1.2, 1.1, 1., 1., 1.1, 1.2, 1., 1.,
                               1., 1., 1.2, 1.1, 1., 1., 1.1, 1.2]])
    target_weight = torch.from_numpy(target_weight).to(device).float()

    # Visdom可视化
    viz = Visdom()
    viz.line([0], [0], win='Train Loss', opts=dict(title='Train Loss'))
    viz.line([0], [0], win='Accuracy', opts=dict(title='Model Accuracy'))

    # 训练循环
    for ep in range(ep_start, num_epochs + 1):
        total_loss, count = 0., 0
        total_accuracy = 0.
        final_accuracy = 0.

        for index, (img, heatmaps, pts) in enumerate(tqdm(data_loader, desc=f'Epoch{ep}')):
            img, heatmaps = img.to(device).float(), heatmaps.to(device).float()

            if use_model in ['ResNet', 'HRNet']:
                heatmaps_pred = model(img)
                loss = criteon(heatmaps_pred, heatmaps, target_weight)
                accuracy = calculate_accuracy(heatmaps_pred, heatmaps)

            elif use_model == 'Hourglass_Stack2':
                heatmaps_preds = model(img)
                heatmaps_pred = heatmaps_preds[-1]
                # 中继监督
                loss1 = criteon(heatmaps_preds[0], heatmaps, target_weight)
                loss2 = criteon(heatmaps_preds[1], heatmaps, target_weight)
                loss = (loss1 + loss2) / 2
                accuracy = calculate_accuracy(heatmaps_pred, heatmaps)

            total_accuracy += accuracy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_step = (ep - 1) * len(data_loader) + index
            total_loss += loss.item()
            count += 1

            if count == 10 or index == len(data_loader) - 1:
                # 可视化损失
                viz.line([total_loss / count], [cur_step], win='Train Loss', update='append')

                # 可视化准确率
                epoch_accuracy = total_accuracy / count
                viz.line([epoch_accuracy], [cur_step], win='Accuracy', update='append')

                viz.image(img[0], win='Image', opts=dict(title='Image'))
                viz.images(heatmaps2rgb(heatmaps[0]), nrow=4,
                           win=f'GT Heatmaps', opts=dict(title=f'GT Heatmaps'))
                viz.images(heatmaps2rgb(heatmaps_pred[0]), nrow=4,
                           win=f'Pred Heatmaps', opts=dict(title=f'red Heatmaps'))

                final_loss = total_loss / count
                final_accuracy = total_accuracy / count
                total_loss, count = 0., 0
                total_accuracy = 0.

        lr_scheduler.step()

        # 保存模型
        torch.save({
            'epoch': ep,
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, f'weights/{use_model}_epoch{ep}_loss{final_loss:.6f}_accuracy{final_accuracy:.4f}.pth')

        torch.cuda.empty_cache()

def main():
    # 通过配置字典灵活设置训练参数
    train_config = {
        'seed': 999,
        'use_model': 'HRNet',
        'lr': 1e-3,
        'batch_size': 8,
        'num_epochs': 10,  # 可以轻松调整训练轮次
        'ckpt': None,
        'num_workers': 8  # 可以轻松调整
    }
    train(train_config)
#210052202019 龙正
if __name__ == '__main__':
    main()