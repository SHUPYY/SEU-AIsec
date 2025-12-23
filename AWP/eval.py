# 导入必要的库
import os  # 操作系统相关功能
import argparse  # 命令行参数解析
import time  # 时间相关功能
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torchvision.datasets as datasets  # 计算机视觉数据集
import torch.utils.data as data  # 数据加载工具
import torchvision.transforms as transforms  # 数据转换工具

import sys

sys.path.insert(0, '..')  # 将上级目录添加到Python路径中

# 导入自定义的模型架构
from AT_AWP.preactresnet import *  # PreAct ResNet模型
from AT_AWP.wideresnet import *  # Wide ResNet模型

# 设置计算设备：如果CUDA可用则使用GPU，否则使用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def filter_state_dict(state_dict):
    """
    过滤和清理模型的状态字典

    功能说明：
    1. 如果状态字典中包含'state_dict'键，则提取其值
    2. 移除包含'sub_block'的键（子模块）
    3. 移除键名中的'module'前缀（通常是DataParallel包装时添加的）

    参数:
        state_dict: 模型的状态字典

    返回:
        过滤后的有序状态字典
    """
    from collections import OrderedDict

    # 如果状态字典嵌套在'state_dict'键中，则提取出来
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # 跳过包含'sub_block'的键
        if 'sub_block' in k:
            continue
        # 移除'module'前缀（DataParallel添加的前缀）
        if 'module' in k:
            new_state_dict[k[7:]] = v  # 去掉'module.'前缀（7个字符）
        else:
            new_state_dict[k] = v
    return new_state_dict


# 图像归一化模块
class Normalize(nn.Module):
    """
    图像归一化模块

    将输入图像按照指定的均值和标准差进行归一化处理
    公式: normalized_x = (x - mean) / std
    """

    def __init__(self, mean, std):
        """
        初始化归一化模块

        参数:
            mean: 各通道的均值，格式为(R, G, B)
            std: 各通道的标准差，格式为(R, G, B)
        """
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)  # 将均值转换为张量
        self.std = torch.tensor(std)  # 将标准差转换为张量

    def forward(self, x):
        """
        前向传播：对输入进行归一化

        参数:
            x: 输入图像张量，shape为[batch, channel, height, width]

        返回:
            归一化后的图像张量
        """
        # 对每个通道进行归一化：(x - mean) / std
        # [None, :, None, None]将均值和标准差扩展为[1, C, 1, 1]的形状，以便广播
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[
            None, :, None, None
        ]


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 模型架构选择：WideResNet28/34或PreActResNet18
    parser.add_argument(
        '--arch',
        type=str,
        default='WideResNet34',
        choices=['WideResNet28', 'WideResNet34', 'PreActResNet18'],
    )
    # 模型检查点文件路径
    parser.add_argument('--checkpoint', type=str, default='./model_test.pt')
    # 数据集选择：CIFAR10或CIFAR100
    parser.add_argument(
        '--data',
        type=str,
        default='CIFAR10',
        choices=['CIFAR10', 'CIFAR100'],
        help='Which dataset the eval is on',
    )
    # 数据集存放目录
    parser.add_argument('--data_dir', type=str, default='./data')
    # 数据预处理方式：meanstd(均值标准差归一化)、01(归一化到[0,1])、+-1(归一化到[-1,1])
    parser.add_argument(
        '--preprocess',
        type=str,
        default='01',
        choices=['meanstd', '01', '+-1'],
        help='The preprocess for data',
    )
    # 攻击范数类型：L2或Linf(无穷范数)
    parser.add_argument('--norm', type=str, default='Linf', choices=['L2', 'Linf'])
    # 攻击扰动大小(epsilon)，默认为8/255
    parser.add_argument('--epsilon', type=float, default=8.0 / 255.0)

    # 测试样本数量，默认10000
    parser.add_argument('--n_ex', type=int, default=10000)
    # 是否单独保存每个对抗样本
    parser.add_argument('--individual', default=False, action='store_true')
    # 对抗样本保存目录
    parser.add_argument('--save_dir', type=str, default='./adv_inputs')
    # 批处理大小
    parser.add_argument('--batch_size', type=int, default=200)
    # 日志文件路径
    parser.add_argument('--log_path', type=str, default='./log.txt')
    # AutoAttack版本：standard(标准版)或custom(自定义版)
    parser.add_argument('--version', type=str, default='standard')

    # 解析命令行参数
    args = parser.parse_args()
    # 从数据集名称中提取类别数（CIFAR10->10, CIFAR100->100）
    num_classes = int(args.data[5:])

    # 根据预处理方式设置归一化参数
    if args.preprocess == 'meanstd':
        # 使用数据集的实际均值和标准差进行归一化
        if args.data == 'CIFAR10':
            # CIFAR10数据集的RGB通道均值和标准差
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif args.data == 'CIFAR100':
            # CIFAR100数据集的RGB通道均值和标准差
            mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    elif args.preprocess == '01':
        # 不进行归一化，数据范围保持在[0, 1]
        mean = (0, 0, 0)
        std = (1, 1, 1)
    elif args.preprocess == '+-1':
        # 归一化到[-1, 1]范围
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError('Please use valid parameters for normalization.')

    # 根据指定架构创建神经网络模型
    # model = ResNet18()
    if args.arch == 'WideResNet34':
        # 创建深度为34、宽度因子为10的Wide ResNet
        net = WideResNet(depth=34, num_classes=num_classes, widen_factor=10)
    elif args.arch == 'WideResNet28':
        # 创建深度为28、宽度因子为10的Wide ResNet
        net = WideResNet(depth=28, num_classes=num_classes, widen_factor=10)
    elif args.arch == 'PreActResNet18':
        # 创建18层的PreAct ResNet
        net = PreActResNet18(num_classes=num_classes)
    else:
        raise ValueError('Please use choose correct architectures.')

    # 加载模型检查点并过滤状态字典
    ckpt = filter_state_dict(torch.load(args.checkpoint, map_location=device))
    # 将权重加载到网络中
    net.load_state_dict(ckpt)

    # 创建完整模型：先归一化，再通过网络
    model = nn.Sequential(Normalize(mean=mean, std=std), net)

    # 将模型移动到指定设备（CPU或GPU）
    model.to(device)
    # 设置为评估模式（关闭dropout、batch normalization等训练特性）
    model.eval()

    # 加载测试数据
    # 定义数据转换：将PIL图像或numpy数组转换为张量
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    # 动态加载指定数据集（CIFAR10或CIFAR100）的测试集
    # root: 数据集存放路径, train=False: 使用测试集, download=True: 如果不存在则下载
    item = getattr(datasets, args.data)(
        root=args.data_dir, train=False, transform=transform_chain, download=True
    )
    # 创建数据加载器：batch_size=1000, 不打乱顺序, 不使用多进程
    test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)

    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 加载AutoAttack攻击方法
    from autoattack import AutoAttack

    # 初始化AutoAttack对象
    # model: 待攻击的模型, norm: 攻击范数类型, eps: 扰动大小, log_path: 日志保存路径
    adversary = AutoAttack(
        model, norm=args.norm, eps=args.epsilon, log_path=args.log_path
    )

    # 将所有测试数据提取并合并成单个张量
    l = [x for (x, y) in test_loader]  # 提取所有图像
    x_test = torch.cat(l, 0)  # 在第0维拼接所有批次的图像
    l = [y for (x, y) in test_loader]  # 提取所有标签
    y_test = torch.cat(l, 0)  # 在第0维拼接所有批次的标签

    # 自定义攻击版本示例
    # cheap version
    # example of custom version
    if args.version == 'custom':
        # 只使用APGD-CE和FAB两种攻击方法
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        # 设置APGD攻击的重启次数为2
        adversary.apgd.n_restarts = 2
        # 设置FAB攻击的重启次数为2
        adversary.fab.n_restarts = 2

    # 运行攻击并保存对抗样本
    # run attack and save images
    if not args.individual:
        # 运行标准评估：生成对抗样本
        # x_test[:args.n_ex]: 使用前n_ex个测试样本
        # y_test[:args.n_ex]: 对应的标签
        # bs: 批处理大小
        adv_complete = adversary.run_standard_evaluation(
            x_test[: args.n_ex], y_test[: args.n_ex], bs=args.batch_size
        )

        # 保存生成的对抗样本
        # 文件名格式: aa_版本_1_样本数_eps_扰动大小.pth
        torch.save(
            {'adv_complete': adv_complete},
            '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
                args.save_dir, 'aa', args.version, adv_complete.shape[0], args.epsilon
            ),
        )
