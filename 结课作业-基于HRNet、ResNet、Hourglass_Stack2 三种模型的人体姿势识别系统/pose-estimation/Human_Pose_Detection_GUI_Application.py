import os
import torch
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
import requests
from io import BytesIO
from PIL import Image as PILImage
import numpy as np

from utils import draw_joints, heatmaps_to_coords
from models.hourglass import hg_stack2
from models.pose_res_net import PoseResNet
from models.hr_net import hr_w32


class PoseDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("人体姿态检测")
        master.geometry("1024x768")

        # 背景设置
        self.setup_gradient_background()

        # 左边框架
        self.frame_left = tk.Frame(master)
        self.frame_left.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # 右边框架
        self.frame_right = tk.Frame(master)
        self.frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 模型选择
        self.model_var = tk.StringVar(value='HRNet')
        self.model_label = tk.Label(self.frame_left, text="选择模型:", bg="white", font=("Arial", 12))
        self.model_label.pack(pady=10)

        self.model_dropdown = ttk.Combobox(self.frame_left, textvariable=self.model_var,
                                           values=['Hourglass_Stack2', 'ResNet', 'HRNet'])
        self.model_dropdown.pack(pady=5)

        # 输入类型选择
        self.input_var = tk.StringVar(value='摄像头')
        self.input_label = tk.Label(self.frame_left, text="选择输入类型:", bg="white", font=("Arial", 12))
        self.input_label.pack(pady=10)

        self.input_dropdown = ttk.Combobox(self.frame_left, textvariable=self.input_var,
                                           values=['摄像头', '图片', '视频'])
        self.input_dropdown.pack(pady=5)

        # 摄像头选择
        self.camera_var = tk.StringVar(value='0')
        self.camera_label = tk.Label(self.frame_left, text="选择摄像头:", bg="white", font=("Arial", 12))
        self.camera_label.pack(pady=10)

        # 动态填充摄像头列表
        self.camera_list = self.get_available_cameras()
        self.camera_dropdown = ttk.Combobox(self.frame_left, textvariable=self.camera_var,
                                            values=self.camera_list)
        self.camera_dropdown.pack(pady=5)

        # 开始按钮
        self.start_button = tk.Button(self.frame_left, text="开始检测", command=self.start_detection, bg="#4CAF50", fg="white", font=("Arial", 12))
        self.start_button.pack(pady=20)

        # 重置按钮
        self.reset_button = tk.Button(self.frame_left, text="重置", command=self.reset_task, bg="#FF5722", fg="white", font=("Arial", 12))
        self.reset_button.pack(pady=10)

        # 保存结果按钮
        self.save_button = tk.Button(self.frame_left, text="保存结果", command=self.save_result, bg="#4CAF50", fg="white", font=("Arial", 12))
        self.save_button.pack(pady=10)

        # 图像显示
        self.image_label = tk.Label(self.frame_right, bg="white", relief=tk.RIDGE, bd=2)
        self.image_label.pack(pady=10, fill=tk.BOTH, expand=True)

        # 模型和设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.ckpt = r'weights\HRNet_epoch20_loss0.000459.pth'

        self.video_generator = None  # 视频生成器，用于更新视频帧

    def setup_gradient_background(self):
        """设置渐变色背景"""
        gradient = Image.new("RGB", (1024, 768), "#4CAF50")
        gradient = gradient.convert("RGB")
        self.bg_image = ImageTk.PhotoImage(gradient)
        self.bg_label = tk.Label(self.master, image=self.bg_image)
        self.bg_label.place(relwidth=1, relheight=1)

    def get_available_cameras(self):
        """检测可用摄像头索引"""
        available_cameras = []
        for i in range(5):  # 检查最多5个摄像头索引
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                available_cameras.append(str(i))
                cap.release()
        return available_cameras

    def load_model(self, model_name):
        """加载所选模型"""
        if model_name == 'Hourglass_Stack2':
            model = hg_stack2().to(self.device)
        elif model_name == 'ResNet':
            model = PoseResNet().to(self.device)
        elif model_name == 'HRNet':
            model = hr_w32().to(self.device)
        else:
            raise NotImplementedError("模型不支持")

        model.load_state_dict(torch.load(self.ckpt)['model'])
        model.eval()
        return model

    def process_frame(self, frame):
        """处理单帧图像进行姿态检测"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (256, 256))

        img = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.

        with torch.no_grad():
            if self.model_var.get() in ['ResNet', 'HRNet']:
                heatmaps_pred = self.model(img)
            elif self.model_var.get() == 'Hourglass_Stack2':
                heatmaps_pred = self.model(img)[-1]

            heatmaps_pred = heatmaps_pred.double()
            heatmaps_pred_np = heatmaps_pred.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            coord_joints = heatmaps_to_coords(heatmaps_pred_np, resolu_out=[256, 256], prob_threshold=0.1)
            img_rgb = draw_joints(img_rgb, coord_joints)

        return img_rgb

    def start_detection(self):
        """根据所选输入类型开始检测"""
        self.model = self.load_model(self.model_var.get())
        input_type = self.input_var.get()
        if input_type == '摄像头':
            self.start_camera_detection()
        elif input_type == '图片':
            self.start_image_detection()
        elif input_type == '视频':
            self.start_video_detection()

    def start_camera_detection(self):
        camera_index = int(self.camera_var.get())
        cap = cv2.VideoCapture(camera_index)

        def update_frame():
            ret, frame = cap.read()
            if ret:
                # 处理图像进行姿态检测
                self.current_frame = self.process_frame(frame)

                # 获取显示区域的大小
                display_width = self.image_label.winfo_width()
                display_height = self.image_label.winfo_height()

                # 获取当前图像的大小
                image_height, image_width = self.current_frame.shape[:2]

                # 等比缩放图像以适应显示框
                scale_width = display_width / image_width
                scale_height = display_height / image_height
                scale = min(scale_width, scale_height)

                # 计算缩放后的图像尺寸
                new_width = int(image_width * scale)
                new_height = int(image_height * scale)

                # 缩放图像
                resized_image = cv2.resize(self.current_frame, (new_width, new_height))

                # 将缩放后的图像转换为适合Tkinter显示的格式
                imgtk = ImageTk.PhotoImage(image=PILImage.fromarray(resized_image))

                # 更新标签显示
                self.image_label.imgtk = imgtk
                self.image_label.configure(image=imgtk)

                # 继续更新帧
                self.image_label.after(10, update_frame)
            else:
                cap.release()
                messagebox.showerror("错误", "无法读取摄像头")

        update_frame()

    def update_video_frame(self):
        try:
            frame = next(self.video_generator)
            # 处理图像进行姿态检测
            self.current_frame = self.process_frame(frame)

            # 获取显示区域的大小
            display_width = self.image_label.winfo_width()
            display_height = self.image_label.winfo_height()

            # 获取当前图像的大小
            image_height, image_width = self.current_frame.shape[:2]

            # 等比缩放图像以适应显示框
            scale_width = display_width / image_width
            scale_height = display_height / image_height
            scale = min(scale_width, scale_height)

            # 计算缩放后的图像尺寸
            new_width = int(image_width * scale)
            new_height = int(image_height * scale)

            # 缩放图像
            resized_image = cv2.resize(self.current_frame, (new_width, new_height))

            # 将缩放后的图像转换为适合Tkinter显示的格式
            imgtk = ImageTk.PhotoImage(image=PILImage.fromarray(resized_image))

            # 更新标签显示
            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)

            # 继续更新视频帧
            self.image_label.after(30, self.update_video_frame)
        except StopIteration:
            messagebox.showinfo("信息", "视频播放完毕")

    def start_image_detection(self):
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        try:
            # 使用 Pillow 加载图片
            img = PILImage.open(file_path)
            img = img.convert("RGB")  # 确保转换为RGB模式
            img = np.array(img)  # 转换为 numpy 数组

            # 将 RGB 转换为 BGR
            img_bgr = img[:, :, ::-1]  # 反转颜色通道

            self.current_frame = self.process_frame(img_bgr)  # 使用正确颜色的图像

            # 获取显示区域的大小
            display_width = self.image_label.winfo_width()
            display_height = self.image_label.winfo_height()

            # 获取当前图像的大小
            image_height, image_width = self.current_frame.shape[:2]

            # 等比缩放图像以适应显示框
            scale_width = display_width / image_width
            scale_height = display_height / image_height
            scale = min(scale_width, scale_height)

            # 计算缩放后的图像尺寸
            new_width = int(image_width * scale)
            new_height = int(image_height * scale)

            # 缩放图像
            resized_image = cv2.resize(self.current_frame, (new_width, new_height))

            # 将缩放后的图像转换为适合Tkinter显示的格式
            imgtk = ImageTk.PhotoImage(image=PILImage.fromarray(resized_image))

            # 更新标签显示
            self.image_label.imgtk = imgtk
            self.image_label.configure(image=imgtk)
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")

    def start_video_detection(self):
        # 获取用户选择的视频文件路径
        video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4;*.avi;*.mkv"), ("All files", "*.*")]
        )
        if not video_path:
            messagebox.showinfo("信息", "未选择视频文件")
            return

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("错误", f"无法打开视频文件：{video_path}")
            return

        def update_frame():
            ret, frame = cap.read()
            if ret:
                # 处理当前帧
                self.current_frame = self.process_frame(frame)

                # 获取显示区域的大小
                display_width = self.image_label.winfo_width()
                display_height = self.image_label.winfo_height()

                # 获取当前帧的大小
                frame_height, frame_width = self.current_frame.shape[:2]

                # 等比缩放帧以适应显示框
                scale_width = display_width / frame_width
                scale_height = display_height / frame_height
                scale = min(scale_width, scale_height)

                # 计算缩放后的尺寸
                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)

                # 缩放当前帧
                resized_frame = cv2.resize(self.current_frame, (new_width, new_height))

                # 转换为适合Tkinter显示的格式
                imgtk = ImageTk.PhotoImage(image=PILImage.fromarray(resized_frame))

                # 更新显示标签
                self.image_label.imgtk = imgtk
                self.image_label.configure(image=imgtk)

                # 继续显示下一帧
                self.image_label.after(30, update_frame)
            else:
                # 视频播放结束
                cap.release()
                messagebox.showinfo("信息", "视频播放完毕")

        update_frame()

    def reset_task(self):
        """重置为刚开始运行的状态"""
        self.image_label.configure(image="")
        self.model = None
        self.video_generator = None
        messagebox.showinfo("信息", "任务已重置")

    def save_result(self):
        if hasattr(self, 'current_frame'):
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
                messagebox.showinfo("信息", f"结果已保存: {file_path}")
        else:
            messagebox.showerror("错误", "没有可保存的结果")

def main():
    root = tk.Tk()
    app = PoseDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
