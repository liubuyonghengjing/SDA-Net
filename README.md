本项目实现了一个用于骨髓细胞图像分类的深度学习框架。

项目结构
text
.
├── model_SDA.py        # 模型定义（TransformerBlock / CASA / ResNeXtFusion 等）
├── train.py            # 训练 & 测试脚本
├── checkpoint_xxx.pth  # 断点续训文件（自动生成）
├── best_val_xxx.pth    # 验证集最佳权重（自动生成）
└── class_indices_xxx.json  # 类别映射表（自动生成）
🖥️ 环境依赖
bash
python >= 3.8
torch >= 1.10
torchvision >= 0.11
scikit-learn
numpy
Pillow
安装：

bash
pip install torch torchvision scikit-learn numpy pillow
