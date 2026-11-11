import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

def process_highres_video():
    video_path = r'./movie_PHI.mp4'
    target_shape = (2300, 3, 224, 224)  # 修正后的CHW维度顺序
    # frame_interval = 50  # 2秒 × 25fps
    # skip_count = 5       # 跳过前10秒（5×2秒）
    frame_interval = 5  # 2秒 × 25fps
    skip_count = 50       # 跳过前10秒（5×2秒）
    
    # 预分配内存（约1.4GB空间）
    video_data = np.zeros(target_shape, dtype=np.float32)
    clip_size = (224,224)
    preproc = transforms.Compose([
            transforms.Resize(size=clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=clip_size),
            # transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"视频文件无法访问：{video_path}")

        # 验证视频基础属性
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_width, actual_height) != (1920, 1080):
            raise ValueError(f"分辨率不匹配！需要1920x1080，实际为{actual_width}x{actual_height}")

        # 定位起始帧（跳过的帧数 = 5间隔 × 50帧/间隔）
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_count * frame_interval)

        # 主处理循环
        frame_count = 0
        with tqdm(total=target_shape[0], desc="视频解析进度") as pbar:
            while frame_count < target_shape[0]:
                ret, frame = cap.read()
                if not ret:
                    break  # 视频提前结束
                # 调整分辨率到720x1280
                # resized_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

                # 维度转换（HWC to CHW）
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame = rgb_frame.transpose(2, 0, 1) / 255.0  # 关键维度调整
                rgb_frame = torch.from_numpy(rgb_frame)
                video_data[frame_count] = preproc(rgb_frame)
                
                # 跳过间隔帧
                for _ in range(frame_interval - 1):
                    cap.grab()
                
                frame_count += 1
                pbar.update(1)

        # 完整性校验
        if frame_count < target_shape[0]:
            raise RuntimeError(f"视频长度不足，仅提取到{frame_count}帧")

    finally:
        cap.release()

    # 保存数据文件
    tensor_data = torch.from_numpy(video_data)
    torch.save(tensor_data, 'image_tensor3.pt')
    print(f"\n处理完成！Tensor形状：{tensor_data.shape}，类型：{tensor_data.dtype}")


if __name__ == "__main__":
    process_highres_video()

'''
import numpy as np
import cv2
import os
from tqdm import tqdm
import torch

def export_frames():
    try:
        # 加载数据文件
        data = torch.load("/home/students/ljx_4090_2/MDD/image_tensor2.pt").to("cpu").numpy() * 255.0
        # data = np.load('image2.npy')
        print("数据加载成功！原始形状：", data.shape)
        
        # 创建输出目录
        os.makedirs('./images2', exist_ok=True)
        
        # 验证数据维度
        if data.shape != (2300, 3, 224, 224):
            raise ValueError("数据形状不符合预期要求")
            
        # 遍历所有帧
        for idx, frame in tqdm(enumerate(data), total=2300, desc="导出进度"):
            # 维度转换 CHW -> HWC
            hwc_frame = frame.transpose(1, 2, 0)
            # 转换颜色空间 RGB -> BGR
            bgr_frame = cv2.cvtColor(hwc_frame, cv2.COLOR_RGB2BGR)
            # 生成文件名
            filename = os.path.join('./images2', f'frame_{idx:04d}.jpg')
            # 保存图像
            cv2.imwrite(filename, bgr_frame)
            
        print("\n导出完成！请检查 ./images 目录下的文件")

    except FileNotFoundError:
        print("错误：未找到 image2.npy 文件")
    except Exception as e:
        print(f"发生错误：{str(e)}")

if __name__ == "__main__":
    export_frames()
'''