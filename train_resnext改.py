import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import json
import os
import time
from model_SDA import resnext50_32x4d
import random
import numpy as np
import warnings
from PIL import Image
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 选择设备
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 处理损坏图像的 ImageFolder 子类
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        for _ in range(10):  # 最多尝试10次，跳过损坏图像
            path, target = self.samples[index]
            try:
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target
            except Exception as e:
                warnings.warn(f"[跳过损坏图像] {path} 错误: {e}")
                index = (index + 1) % len(self.samples)
        raise RuntimeError("连续跳过 10 张图像，可能数据集有大量损坏图像")

# 数据增强
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.67, 0.54, 0.85), (0.19, 0.29, 0.08)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.67, 0.54, 0.85), (0.19, 0.29, 0.08))
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.67, 0.54, 0.85), (0.19, 0.29, 0.08))
    ])
}

# 数据路径
image_path = '/mnt/data/zhl/cells-19-180000/'#/mnt/data/zhl/BM_cytomorphology_data/#/mnt/data/zhl/cells-19-180000/#/mnt/data/zhl/BMCD-19/#/mnt/data/zhl/PBC-8/

# 加载数据集
train_dataset = SafeImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
validate_dataset = SafeImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
test_dataset = SafeImageFolder(root=os.path.join(image_path, "test"), transform=data_transform["test"])

# 统计数据集大小
train_num = len(train_dataset)
val_num = len(validate_dataset)
test_num = len(test_dataset)

print(f"Train set size: {train_num}")
print(f"Validation set size: {val_num}")
print(f"Test set size: {test_num}")

# 保存类别标签
class_dict = train_dataset.class_to_idx
with open('class_indices_cells-19.json', 'w') as f:#class_indices_BM-16.json#class_indices_cells-19.json#class_indices_BMCD-19.json
    json.dump(class_dict, f, indent=4)



# DataLoader
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 初始化模型
net = resnext50_32x4d(num_classes=len(class_dict), include_top=True).to(device)

# if torch.cuda.device_count() > 1:
#     net = torch.nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net, device_ids=[2])

# 损失函数与优化器
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003, weight_decay=1e-4)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=15,
    verbose=True,
    min_lr=1e-6
)

checkpoint_path = 'checkpoint_resume_ci-fangcha.pth'
# 保存路径与训练参数
save_path_val_best = 'CSAMBL214-fangcha-Resnext50_val_best_cells-19-32.pth'
#save_path_test_best = 'shuangWNResnext50_test_best_BM.pth'
best_val_acc = 0.0
best_test_acc = 0.0
num_epochs = 500
patience = 50
patience_counter = 0

start_epoch = 0

# === 如果有 checkpoint 就加载 ===
if os.path.exists(checkpoint_path):
    print("=> Loading checkpoint to resume training...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint['best_val_acc']
    #best_test_acc = checkpoint['best_test_acc']
    #patience_counter = checkpoint.get('patience_counter', 0)
    print(f"=> Resuming from epoch {start_epoch}")
else:
    print("=> No checkpoint found. Starting from scratch.")

# 训练过程
for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()

    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()

        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()

        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * (50 - len(a))
        print(f"\rtrain loss: {int(rate * 100):^3d}% [{a}->{b}] {loss:.3f}", end="")

    print()
    print(f"Epoch {epoch + 1} training time: {time.perf_counter() - t1:.2f}s")

    # 验证集评估
    net.eval()
    val_acc = 0.0
    val_loss = 0.0
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            loss = loss_function(outputs, val_labels.to(device))
            val_loss += loss.item()
            predict_y = torch.max(outputs, dim=1)[1]
            val_acc += (predict_y == val_labels.to(device)).sum().item()
    val_accuracy = val_acc / val_num
    avg_val_loss = val_loss / len(validate_loader)

    # # 测试集评估
    # test_acc = 0.0
    # with torch.no_grad():
    #     for test_data in test_loader:
    #         test_images, test_labels = test_data
    #         outputs = net(test_images.to(device))
    #         predict_y = torch.max(outputs, dim=1)[1]
    #         test_acc += (predict_y == test_labels.to(device)).sum().item()
    # test_accuracy = test_acc / test_num

    # 更新学习率
    # scheduler.step(val_accuracy)
    # current_lr = optimizer.param_groups[0]['lr']
    # print(f"Current learning rate: {current_lr:.2e}")
    # print(f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f}  val_accuracy: {val_accuracy:.3f}  test_accuracy: {test_accuracy:.3f}')
    scheduler.step(val_accuracy)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr:.2e}")
    print(f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f}  val_accuracy: {val_accuracy:.3f}')
    # 模型保存与早停
    # has_improved = False
    #
    # if val_accuracy > best_val_acc:
    #     best_val_acc = val_accuracy
    #     torch.save(net.state_dict(), save_path_val_best)
    #     print(f"Validation best model updated: {best_val_acc:.3f}")
    #     has_improved = True

    # has_improved = False
    #
    # if val_accuracy > best_val_acc:
    #     best_val_acc = val_accuracy
    #     torch.save(net.state_dict(), save_path_val_best)
    #     print(f"Validation best model updated: {best_val_acc:.3f}")
    #     has_improved = True
    #
    # if has_improved:
    #     patience_counter = 0
    # else:
    #     patience_counter += 1
    #
    # if patience_counter >= patience:
    #     print(f"Early stopping triggered after {patience} epochs of no improvement.")
    #     break
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(net.state_dict(), save_path_val_best)
        print(f"Best model updated with accuracy: {best_val_acc:.3f}")
        patience_counter = 0
    else:
        patience_counter += 1

    # 早停机制
    if patience_counter >= patience:
        print(f"Early stopping triggered after {patience} epochs of no improvement.")
        break
    # if test_accuracy > best_test_acc:
    #     best_test_acc = test_accuracy
    #     torch.save(net.state_dict(), save_path_test_best)
    #     print(f"Test best model updated: {best_test_acc:.3f}")
    #     has_improved = True
    #
    # if has_improved:
    #     patience_counter = 0
    # else:
    #     patience_counter += 1
    #
    # if patience_counter >= patience:
    #     print(f"Early stopping triggered after {patience} epochs of no improvement.")
    #     break

        # === 每个 epoch 保存中断用 checkpoint ===
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        #'best_test_acc': best_test_acc,
        'patience_counter': patience_counter,
    }, checkpoint_path)

# print("Finished Training.")
# print("=" * 60)
# print(f"Best validation accuracy: {best_val_acc:.4f}")
# print(f"Best test accuracy:       {best_test_acc:.4f}")
# print("=" * 60)
# === 修改点：训练完成后再测试 ===
# print("=" * 60)
# print("Evaluating on test set with best validation model...")
# net.load_state_dict(torch.load(save_path_val_best))
# net.eval()
# test_acc = 0.0
# with torch.no_grad():
#     for test_data in test_loader:
#         test_images, test_labels = test_data
#         outputs = net(test_images.to(device))
#         predict_y = torch.max(outputs, dim=1)[1]
#         test_acc += (predict_y == test_labels.to(device)).sum().item()
# test_accuracy = test_acc / test_num
# print(f"Final test accuracy (using best val model): {test_accuracy:.4f}")
print("=" * 60)
print("Evaluating on test set with best validation model...")

# 载入最佳模型
net.load_state_dict(torch.load(save_path_val_best, map_location=device))
net.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for test_images, test_labels in test_loader:
        outputs = net(test_images.to(device))
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels = test_labels.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

# 转成 numpy
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# 计算指标
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"Test Accuracy : {acc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall   : {recall:.4f}")
print(f"Test F1-score : {f1:.4f}")

# 输出详细分类报告（保留四位小数）
print("\nDetailed classification report:")
print(classification_report(
    all_labels,
    all_preds,
    target_names=list(class_dict.keys()),
    zero_division=0,
    digits=4
))
