# import argparse
# import logging
# import sys
# import time
# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os

# from wideresnet import WideResNet
# from preactresnet import PreActResNet18
# from utils import *
# from utils_awp import AdvWeightPerturb

# mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
# std = torch.tensor(cifar10_std).view(3,1,1).cuda()

# def normalize(X):
#     return (X - mu)/std

# upper_limit, lower_limit = 1,0

# def clamp(X, lower_limit, upper_limit):
#     return torch.max(torch.min(X, upper_limit), lower_limit)

# class Batches():
#     def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.set_random_choices = set_random_choices
#         self.dataloader = torch.utils.data.DataLoader(
#             dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
#         )

#     def __iter__(self):
#         if self.set_random_choices:
#             self.dataset.set_random_choices()
#         return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

#     def __len__(self):
#         return len(self.dataloader)

# def attack_pgd(model, X, y, epsilon, alpha, attack_iters, norm='l_inf'):
#     delta = torch.zeros_like(X).cuda()
#     delta.uniform_(-epsilon, epsilon)
#     delta = clamp(delta, lower_limit - X, upper_limit - X)
#     delta.requires_grad = True

#     for _ in range(attack_iters):
#         output = model(normalize(X + delta))
#         loss = F.cross_entropy(output, y)
#         loss.backward()

#         grad = delta.grad.detach()
#         if norm == "l_inf":
#             delta = delta + alpha * torch.sign(grad)

#         delta = torch.clamp(delta, -epsilon, epsilon)
#         delta = clamp(delta, lower_limit - X, upper_limit - X)

#         # 关键修复：in-place detach + requires_grad
#         delta = delta.detach_().requires_grad_()

#     return delta.detach()

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epochs', default=30, type=int)
#     parser.add_argument('--batch-size', default=64, type=int)
#     parser.add_argument('--batch-size-test', default=128, type=int)
#     parser.add_argument('--data-dir', default='./cifar-data', type=str)
#     parser.add_argument('--lr-max', default=0.1, type=float)
#     parser.add_argument('--attack-iters', default=5, type=int)
#     parser.add_argument('--attack-iters-test', default=5, type=int)
#     parser.add_argument('--epsilon', default=8, type=int)
#     parser.add_argument('--pgd-alpha', default=2, type=float)
#     parser.add_argument('--awp-gamma', default=0.01, type=float)
#     parser.add_argument('--awp-warmup', default=10, type=int)
#     parser.add_argument('--no-awp', action='store_true', help='Run pure AT without AWP')
#     parser.add_argument('--save-dir', default='../kaggle_exp/baseline/', type=str)
#     parser.add_argument('--seed', default=0, type=int)
#     return parser.parse_args()

# def main():
#     args = get_args()
    
#     if args.no_awp:
#         args.awp_warmup = np.infty
#         method_name = 'AT'
#     else:
#         method_name = 'AT_AWP'

#     save_dir = os.path.join(args.save_dir, method_name)
#     os.makedirs(save_dir, exist_ok=True)

#     logger = logging.getLogger(__name__)
#     logging.basicConfig(
#         format='[%(asctime)s] - %(message)s',
#         datefmt='%Y/%m/%d %H:%M:%S',
#         level=logging.DEBUG,
#         handlers=[
#             logging.FileHandler(os.path.join(save_dir, 'log.txt')),
#             logging.StreamHandler(sys.stdout)
#         ])

#     logger.info(args)

#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed(args.seed)

#     dataset = cifar10(args.data_dir)
#     train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.), dataset['train']['labels']))
#     train_set_x = Transform(train_set, [Crop(32, 32), FlipLR()])
#     train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

#     test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
#     test_batches = Batches(test_set, args.batch_size_test, shuffle=False, num_workers=2)

#     epsilon = (args.epsilon / 255.)
#     pgd_alpha = (args.pgd_alpha / 255.)

#     model = PreActResNet18()
#     proxy = PreActResNet18()
#     model = nn.DataParallel(model).cuda()
#     proxy = nn.DataParallel(proxy).cuda()

#     opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
#     proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
#     awp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=args.awp_gamma)

#     criterion = nn.CrossEntropyLoss()

#     def lr_schedule(t):
#         return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))

#     logger.info('Epoch \t LR \t Train Acc \t Train Robust Acc \t Test Clean Acc \t Test PGD-5 Acc')

#     for epoch in range(args.epochs):
#         model.train()
#         lr = lr_schedule(epoch)
#         opt.param_groups[0].update(lr=lr)

#         train_acc = 0
#         train_robust_acc = 0
#         train_n = 0

#         for i, batch in enumerate(train_batches):
#             X, y = batch['input'], batch['target']

#             delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters)
#             X_adv = normalize(torch.clamp(X + delta, min=lower_limit, max=upper_limit))

#             if epoch >= args.awp_warmup:
#                 awp = awp_adversary.calc_awp(inputs_adv=X_adv, targets=y)
#                 awp_adversary.perturb(awp)

#             output_adv = model(X_adv)
#             loss_adv = criterion(output_adv, y)

#             opt.zero_grad()
#             loss_adv.backward()
#             opt.step()

#             if epoch >= args.awp_warmup:
#                 awp_adversary.restore(awp)

#             output_clean = model(normalize(X))
#             train_robust_acc += (output_adv.max(1)[1] == y).sum().item()
#             train_acc += (output_clean.max(1)[1] == y).sum().item()
#             train_n += y.size(0)

#         # Test
#         model.eval()
#         test_clean_correct = 0
#         test_robust_correct = 0
#         test_n = 0

#         for batch in test_batches:
#             X, y = batch['input'], batch['target']
#             delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test)
#             X_adv = normalize(torch.clamp(X + delta, min=lower_limit, max=upper_limit))

#             with torch.no_grad():
#                 output_clean = model(normalize(X))
#                 output_adv = model(X_adv)

#             test_clean_correct += (output_clean.max(1)[1] == y).sum().item()
#             test_robust_correct += (output_adv.max(1)[1] == y).sum().item()
#             test_n += y.size(0)

#         clean_acc = 100. * test_clean_correct / test_n
#         robust_acc = 100. * test_robust_correct / test_n

#         logger.info('%d \t %.4f \t %.2f \t %.2f \t %.2f \t %.2f',
#                     epoch, lr, 100.*train_acc/train_n, 100.*train_robust_acc/train_n, clean_acc, robust_acc)

#         if epoch == args.epochs - 1:
#             torch.save(model.state_dict(), os.path.join(save_dir, 'final.pth'))

#     print(f"{method_name} 训练完成！最终 Test Clean: {clean_acc:.2f}%, PGD-5: {robust_acc:.2f}%")

# if __name__ == "__main__":
#     main()
import argparse
import logging
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# =========================
# 模型结构导入
# =========================
from wideresnet import WideResNet
from preactresnet import PreActResNet18

# =========================
# 工具函数与 AWP 模块
# =========================
from utils import *
from utils_awp import AdvWeightPerturb

# =========================
# CIFAR-10 数据归一化参数
# =========================
mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

# 对输入进行标准化（与测试保持一致）
def normalize(X):
    return (X - mu)/std

# 输入像素取值范围
upper_limit, lower_limit = 1,0

# 对输入进行裁剪，确保像素合法
def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

# =========================
# 数据加载封装（与 Madry 风格一致）
# =========================
class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        # 关键点：每个 epoch 重新采样数据增强参数（Crop / Flip）
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)

# =========================
# PGD 攻击（L_inf，训练 / 测试共用）
# =========================
def attack_pgd(model, X, y, epsilon, alpha, attack_iters, norm='l_inf'):
    # 初始化扰动
    delta = torch.zeros_like(X).cuda()
    delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True

    for _ in range(attack_iters):
        output = model(normalize(X + delta))
        loss = F.cross_entropy(output, y)
        loss.backward()

        grad = delta.grad.detach()
        if norm == "l_inf":
            delta = delta + alpha * torch.sign(grad)

        delta = torch.clamp(delta, -epsilon, epsilon)
        delta = clamp(delta, lower_limit - X, upper_limit - X)

        # ===== 关键修复说明 =====
        # 该 in-place detach 是为了避免计算图不断增长
        # 否则长时间训练会出现显存泄漏
        delta = delta.detach_().requires_grad_()

    return delta.detach()

# =========================
# 参数解析
# =========================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--batch-size-test', default=128, type=int)
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--attack-iters', default=5, type=int)
    parser.add_argument('--attack-iters-test', default=5, type=int)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--awp-gamma', default=0.01, type=float)
    parser.add_argument('--awp-warmup', default=10, type=int)
    parser.add_argument('--no-awp', action='store_true', help='Run pure AT without AWP')
    parser.add_argument('--save-dir', default='../kaggle_exp/baseline/', type=str)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()

# =========================
# 主训练流程
# =========================
def main():
    args = get_args()
    
    # 是否启用 AWP
    if args.no_awp:
        args.awp_warmup = np.infty
        method_name = 'AT'
    else:
        method_name = 'AT_AWP'

    save_dir = os.path.join(args.save_dir, method_name)
    os.makedirs(save_dir, exist_ok=True)

    # =========================
    # 日志系统（文件 + 终端）
    # =========================
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(save_dir, 'log.txt')),
            logging.StreamHandler(sys.stdout)
        ])

    logger.info(args)

    # 固定随机种子，保证可复现性
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # =========================
    # CIFAR-10 数据集
    # =========================
    dataset = cifar10(args.data_dir)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.), dataset['train']['labels']))
    train_set_x = Transform(train_set, [Crop(32, 32), FlipLR()])
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size_test, shuffle=False, num_workers=2)

    # 攻击强度（归一化）
    epsilon = (args.epsilon / 255.)
    pgd_alpha = (args.pgd_alpha / 255.)

    # =========================
    # 主模型 + proxy 模型（AWP 核心）
    # =========================
    model = PreActResNet18()
    proxy = PreActResNet18()
    model = nn.DataParallel(model).cuda()
    proxy = nn.DataParallel(proxy).cuda()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
    proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)

    # ===== AWP 对抗器 =====
    # Layer-wise AWP 的真实“改进实现”在 utils_awp.py 内部
    awp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=args.awp_gamma)

    criterion = nn.CrossEntropyLoss()

    # 余弦退火学习率
    def lr_schedule(t):
        return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))

    logger.info('Epoch \t LR \t Train Acc \t Train Robust Acc \t Test Clean Acc \t Test PGD-5 Acc')

    # =========================
    # 训练循环
    # =========================
    for epoch in range(args.epochs):
        model.train()
        lr = lr_schedule(epoch)
        opt.param_groups[0].update(lr=lr)

        train_acc = 0
        train_robust_acc = 0
        train_n = 0

        for i, batch in enumerate(train_batches):
            X, y = batch['input'], batch['target']

            # Step 1: 生成输入对抗样本
            delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters)
            X_adv = normalize(torch.clamp(X + delta, min=lower_limit, max=upper_limit))

            # Step 2: AWP（权重对抗扰动）
            if epoch >= args.awp_warmup:
                awp = awp_adversary.calc_awp(inputs_adv=X_adv, targets=y)
                awp_adversary.perturb(awp)

            # Step 3: 使用扰动权重进行反向传播
            output_adv = model(X_adv)
            loss_adv = criterion(output_adv, y)

            opt.zero_grad()
            loss_adv.backward()
            opt.step()

            # Step 4: 恢复权重
            if epoch >= args.awp_warmup:
                awp_adversary.restore(awp)

            # 统计训练精度
            output_clean = model(normalize(X))
            train_robust_acc += (output_adv.max(1)[1] == y).sum().item()
            train_acc += (output_clean.max(1)[1] == y).sum().item()
            train_n += y.size(0)

        # =========================
        # 测试阶段（Clean + PGD-5）
        # =========================
        model.eval()
        test_clean_correct = 0
        test_robust_correct = 0
        test_n = 0

        for batch in test_batches:
            X, y = batch['input'], batch['target']
            delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test)
            X_adv = normalize(torch.clamp(X + delta, min=lower_limit, max=upper_limit))

            with torch.no_grad():
                output_clean = model(normalize(X))
                output_adv = model(X_adv)

            test_clean_correct += (output_clean.max(1)[1] == y).sum().item()
            test_robust_correct += (output_adv.max(1)[1] == y).sum().item()
            test_n += y.size(0)

        clean_acc = 100. * test_clean_correct / test_n
        robust_acc = 100. * test_robust_correct / test_n

        logger.info('%d \t %.4f \t %.2f \t %.2f \t %.2f \t %.2f',
                    epoch, lr, 100.*train_acc/train_n, 100.*train_robust_acc/train_n, clean_acc, robust_acc)

        # 仅保存最终模型（论文复现实验常规做法）
        if epoch == args.epochs - 1:
            torch.save(model.state_dict(), os.path.join(save_dir, 'final.pth'))

    print(f"{method_name} 训练完成！最终 Test Clean: {clean_acc:.2f}%, PGD-5: {robust_acc:.2f}%")

if __name__ == "__main__":
    main()