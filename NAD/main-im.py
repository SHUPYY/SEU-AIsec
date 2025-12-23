"""
main-im.py

后门防御实验入口文件

本文件基于 main.py 修改而来，用于构建完整的后门防御实验框架。
主要功能包括：
1. 解析命令行参数并初始化实验环境
2. 加载教师模型与学生模型，构建蒸馏防御框架
3. 调用后门防御算法（如 ARGD）对学生模型进行训练
4. 在干净数据与后门数据上分别评估模型性能（ACC / ASR）
5. 记录实验结果并保存最优模型

该文件对原 main.py 中的若干错误进行了修复，并针对后门防御任务
对训练与测试流程进行了完善。

作者：王小康

"""

from torch import nn
from models.selector import *
from utils.util import *
from data_loader import get_train_loader, get_test_loader
from at import AT
from config import get_arguments
from argDistillationLoss import argDL
import datetime


def unique_shape(s_shapes):
    """
    提取特征图形状列表中的唯一形状

    Args:
        s_shapes: 特征图形状列表，例如 [(C1,H1,W1), (C2,H2,W2), ...]

    Returns:
        n_s: 每个形状对应的唯一形状索引列表
        unique_shapes: 去重后的形状列表

    示例：
        输入: [(16,32,32), (32,16,16), (16,32,32)]
        输出: [0, 1, 0], [(16,32,32), (32,16,16)]
    """
    n_s = []  # 存储每个形状对应的唯一索引
    unique_shapes = []  # 存储唯一的形状
    n = -1  # 唯一形状的计数器

    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            # 如果是新的形状，添加到唯一形状列表中
            unique_shapes.append(s_shape)
            n += 1
        # 记录当前形状对应的唯一索引
        n_s.append(n)

    return n_s, unique_shapes


def train_step(opt, train_loader, nets, optimizer, criterions, epoch):
    """
    执行一个epoch的训练步骤

    Args:
        opt: 配置参数对象
        train_loader: 训练数据加载器
        nets: 包含学生网络和教师网络的字典
        optimizer: 优化器
        criterions: 包含各种损失函数的字典
        epoch: 当前训练轮次
    """
    # 初始化平均值记录器，用于跟踪损失和精度
    losses = AverageMeter()  # 总损失
    top1 = AverageMeter()  # Top-1准确率
    top5 = AverageMeter()  # Top-5准确率

    # 获取学生网络和教师网络
    snet = nets['snet']  # 学生网络（需要训练的模型）
    tnet = nets['tnet']  # 教师网络（固定参数，用于蒸馏）

    # 获取各种损失函数
    criterionCls = criterions['criterionCls']  # 分类损失（交叉熵）
    criterionAT = criterions['criterionAT']  # 注意力转移损失
    criterionargDL = criterions['criterionargDL']  # ARGD蒸馏损失

    # 将学生网络设置为训练模式
    snet.train()

    # 遍历训练数据批次
    for idx, (img, target) in enumerate(train_loader, start=1):
        # 将数据移到GPU（如果可用）
        if opt.cuda:
            img = img.cuda()  # 输入图像
            target = target.cuda()  # 真实标签

        # 学生网络前向传播，获取多层特征和输出
        # activation1/2/3_s: 学生网络的三层中间特征图
        # output_s: 学生网络的最终输出（分类logits）
        activation1_s, activation2_s, activation3_s, output_s = snet(img)

        # 教师网络前向传播（不需要计算梯度）
        with torch.no_grad():
            activation1_t, activation2_t, activation3_t, output_t = tnet(img)

        # 计算分类损失：学生网络输出与真实标签的交叉熵
        cls_loss = criterionCls(output_s, target)

        # 计算ARGD蒸馏损失：学生特征图向教师特征图对齐
        # 使用三层特征图进行知识蒸馏
        ARG_loss = criterionargDL(
            [activation1_s, activation2_s, activation3_s],  # 学生网络的特征
            [
                activation1_t.detach(),  # 教师网络的特征（已分离梯度）
                activation2_t.detach(),
                activation3_t.detach(),
            ],
        )

        # 总损失 = 分类损失 + ARGD蒸馏损失
        ARG_loss = cls_loss + ARG_loss

        loss_sum = ARG_loss

        # 计算准确率指标
        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))  # Top-1和Top-5准确率

        # 更新平均值记录器
        losses.update(loss_sum.item(), img.size(0))  # 更新损失的平均值
        top1.update(prec1.item(), img.size(0))  # 更新Top-1准确率
        top5.update(prec5.item(), img.size(0))  # 更新Top-5准确率

        # 反向传播和参数更新
        optimizer.zero_grad()  # 清空梯度
        loss_sum.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        # 定期打印训练进度
        if idx % opt.print_freq == 0:
            print(
                'Epoch[{0}]:[{1:03}/{2:03}] '
                'AT_loss:{losses.val:.4f}({losses.avg:.4f})  '
                'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
                    epoch, idx, len(train_loader), losses=losses, top1=top1, top5=top5
                )
            )


def test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch):
    """
    测试模型性能，分别在干净数据和后门数据上评估

    Args:
        opt: 配置参数对象
        test_clean_loader: 干净测试数据加载器
        test_bad_loader: 后门测试数据加载器
        nets: 包含学生网络和教师网络的字典
        criterions: 包含各种损失函数的字典
        epoch: 当前训练轮次

    Returns:
        acc_clean: 在干净数据上的准确率 [top1, top5]
        acc_bd: 在后门数据上的准确率和损失 [top1, top5, cls_loss, at_loss]
    """
    test_process = []  # 用于记录测试过程
    top1 = AverageMeter()  # Top-1准确率记录器
    top5 = AverageMeter()  # Top-5准确率记录器

    # 获取学生网络和教师网络
    snet = nets['snet']
    tnet = nets['tnet']

    # 获取损失函数
    criterionCls = criterions['criterionCls']  # 分类损失
    criterionAT = criterions['criterionAT']  # 注意力转移损失

    # 设置为评估模式（关闭dropout和batch normalization的训练行为）
    snet.eval()
    tnet.eval()

    # ========== 第一阶段：测试干净样本的准确率 ==========
    # 遍历干净测试数据
    for idx, (img, target) in enumerate(test_clean_loader, start=1):
        img = img.cuda()  # 移到GPU
        target = target.cuda()

        # 前向传播（不计算梯度）
        with torch.no_grad():
            # 只需要最终输出，忽略中间特征图
            _, _, _, output_s = snet(img)

        # 计算准确率
        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    # 保存干净样本的测试结果
    acc_clean = [top1.avg, top5.avg]

    # ========== 第二阶段：测试后门样本的攻击成功率 ==========
    # 重新初始化记录器
    cls_losses = AverageMeter()  # 分类损失记录器
    at_losses = AverageMeter()  # 注意力转移损失记录器
    top1 = AverageMeter()  # Top-1准确率记录器
    top5 = AverageMeter()  # Top-5准确率记录器

    # 遍历后门测试数据
    for idx, (img, target) in enumerate(test_bad_loader, start=1):
        img = img.cuda()  # 移到GPU
        target = target.cuda()

        # 前向传播（不计算梯度）
        with torch.no_grad():
            # 获取学生网络的中间特征和输出
            activation1_s, activation2_s, activation3_s, output_s = snet(img)
            # 获取教师网络的中间特征（用于计算AT损失）
            activation1_t, activation2_t, activation3_t, _ = tnet(img)

            # 计算三层特征图的注意力转移损失
            at3_loss = criterionAT(activation3_s, activation3_t).detach()  # 第3层AT损失
            at2_loss = criterionAT(activation2_s, activation2_t).detach()  # 第2层AT损失
            at1_loss = criterionAT(activation1_s, activation1_t).detach()  # 第1层AT损失
            at_loss = at3_loss + at2_loss + at1_loss  # 总AT损失

            # 计算分类损失
            cls_loss = criterionCls(output_s, target)

        # 计算准确率（这里的准确率反映了攻击成功率ASR）
        prec1, prec5 = accuracy(output_s, target, topk=(1, 5))

        # 更新各项指标的平均值
        cls_losses.update(cls_loss.item(), img.size(0))
        at_losses.update(at_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    # 保存后门样本的测试结果
    # top1/top5: 攻击成功率（ASR），cls_losses: 分类损失，at_losses: AT损失
    acc_bd = [top1.avg, top5.avg, cls_losses.avg, at_losses.avg]

    # 打印测试结果
    print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))  # 干净样本的Top-1准确率
    print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))  # 后门样本的Top-1准确率（ASR）

    # 将测试结果保存到CSV文件
    log_root = opt.log_root + '/results.csv'
    # 添加当前epoch的测试数据
    test_process.append((epoch, acc_clean[0], acc_bd[0], acc_bd[2], acc_bd[3]))
    # 创建DataFrame
    df = pd.DataFrame(
        test_process,
        columns=(
            "epoch",  # 训练轮次
            "test_clean_acc",  # 干净数据准确率
            "test_bad_acc",  # 后门数据准确率（ASR）
            "test_bad_cls_loss",  # 后门数据的分类损失
            "test_bad_at_loss",  # 后门数据的AT损失
        ),
    )
    # 以追加模式写入CSV文件
    df.to_csv(log_root, mode='a', index=False, encoding='utf-8')

    return acc_clean, acc_bd


def train(opt):
    """
    主训练函数：执行完整的后门防御训练流程

    流程：
    1. 加载教师模型和学生模型
    2. 获取特征图形状信息
    3. 初始化优化器和损失函数
    4. 加载训练和测试数据
    5. 执行多轮训练和测试
    6. 保存最佳模型

    Args:
        opt: 配置参数对象，包含所有训练相关的超参数
    """
    # ========== 第一步：加载模型 ==========
    print('----------- Network Initialization --------------')
    # 教师模型和学生模型的权重文件路径
    teacher_path = "./weight/t_net/WRN-16-1-T-model_best.pth.tar"
    student_path = "./weight/s_net/WRN-16-1-S-model_best.pth.tar"

    # 加载教师模型（WRN-16-1：Wide ResNet，深度16，宽度因子1）
    teacher = select_model(
        dataset=opt.data_name,  # 数据集名称（如CIFAR-10）
        model_name='WRN-16-1',  # 模型架构
        pretrained=True,  # 使用预训练权重
        pretrained_models_path=teacher_path,  # 权重文件路径
        n_classes=opt.num_class,  # 分类类别数
    ).to(
        opt.device
    )  # 移到指定设备（CPU或GPU）

    print('finished teacher model init...')

    # 加载学生模型（结构与教师相同，但权重来自后门模型）
    student = select_model(
        dataset=opt.data_name,
        model_name='WRN-16-1',
        pretrained=True,
        pretrained_models_path=student_path,
        n_classes=opt.num_class,
    ).to(opt.device)

    print('finished student model init...')
    # 初始化注意力转移损失（参数2.0是温度系数）
    criterionAT = AT(2.0)
    # 将教师模型设置为评估模式（固定参数，不参与训练）
    teacher.eval()

    # ========== 第二步：获取特征图形状信息 ==========
    # 通过前向传播一个随机输入，获取网络中间层特征图的形状
    # 这些形状信息用于初始化ARGD损失函数中的特征对齐模块
    opt.image_size = 32  # CIFAR-10图像大小为32x32
    # 创建一个随机输入张量 [batch_size=1, channels=3, height=32, width=32]
    data = torch.randn(1, 3, opt.image_size, opt.image_size).to(opt.device)
    teacher.eval()  # 确保处于评估模式
    student.eval()

    # 进行前向传播获取特征图
    with torch.no_grad():
        # 教师网络的三层特征图和最终输出
        activation1_t, activation2_t, activation3_t, _ = teacher(data)
        # 学生网络的三层特征图和最终输出
        activation1_s, activation2_s, activation3_s, _ = student(data)

    # s_shapes 和 t_shapes：存储原始特征图的形状（[C, H, W]）
    # 用于 LinearTransform 中的 channel_mean 输入维度
    # 例如：activation1_s.size() = [1, 16, 32, 32]
    #       activation1_s.size()[1:] = [16, 32, 32] = (channels, height, width)
    opt.s_shapes = [
        activation1_s.size()[1:],  # 学生网络第1层特征图形状
        activation2_s.size()[1:],  # 学生网络第2层特征图形状
        activation3_s.size()[1:],  # 学生网络第3层特征图形状
    ]

    opt.t_shapes = [
        activation1_t.size()[1:],  # 教师网络第1层特征图形状
        activation2_t.size()[1:],  # 教师网络第2层特征图形状
        activation3_t.size()[1:],  # 教师网络第3层特征图形状
    ]

    # unique_t_shapes：用于 Sample 模块的 AdaptiveAvgPool2d
    # 由于教师和学生都是 WRN-16-1，空间尺寸相同，所以直接使用 t_shapes
    # n_t: 每个形状对应的唯一索引列表
    # unique_t_shapes: 去重后的形状列表
    opt.n_t, opt.unique_t_shapes = unique_shape(opt.t_shapes)

    # 打印形状信息用于调试
    print(f"Student shapes: {opt.s_shapes}")
    print(f"Teacher shapes: {opt.t_shapes}")
    print(f"Unique teacher shapes for sampling: {opt.unique_t_shapes}")

    # 将网络组织成字典，方便传递
    nets = {'snet': student, 'tnet': teacher}

    # 冻结教师网络的所有参数（不参与训练）
    for param in teacher.parameters():
        param.requires_grad = False

    # ========== 第三步：初始化优化器 ==========
    # 使用SGD优化器，仅优化学生网络的参数
    optimizer = torch.optim.SGD(
        student.parameters(),  # 只优化学生网络
        lr=opt.lr,  # 学习率
        momentum=opt.momentum,  # 动量系数
        weight_decay=opt.weight_decay,  # L2正则化系数
        nesterov=True,  # 使用Nesterov动量
    )

    # ========== 第四步：定义损失函数 ==========
    if opt.cuda:
        criterionCls = nn.CrossEntropyLoss().cuda()  # 分类损失（交叉熵）
        criterionAT = AT(opt.p)  # 注意力转移损失
        criterionargDL = argDL(opt)  # ARGD蒸馏损失
    else:
        criterionCls = nn.CrossEntropyLoss()
        criterionAT = AT(opt.p)
        criterionargDL = argDL(opt)

    # ========== 第五步：加载数据 ==========
    print('----------- DATA Initialization --------------')
    train_loader = get_train_loader(opt)  # 训练数据加载器（包含后门样本）
    # 获取测试数据加载器：干净数据和后门数据
    test_clean_loader, test_bad_loader = get_test_loader(opt)

    # ========== 第六步：开始训练 ==========
    print('----------- Train Initialization --------------')
    for epoch in range(0, opt.epochs):  # 遍历所有训练轮次

        # 根据当前epoch调整学习率（学习率衰减策略）
        adjust_learning_rate(optimizer, epoch, opt.lr)

        # 将所有损失函数组织成字典
        criterions = {
            'criterionCls': criterionCls,  # 分类损失
            'criterionAT': criterionAT,  # AT损失
            'criterionargDL': criterionargDL,  # ARGD蒸馏损失
        }

        # 在第一个epoch之前，先进行一次测试（评估初始性能）
        if epoch == 0:
            test(opt, test_clean_loader, test_bad_loader, nets, criterions, epoch)

        # 执行一个epoch的训练
        train_step(opt, train_loader, nets, optimizer, criterions, epoch + 1)

        # 训练完成后进行测试
        print('testing the models......')
        acc_clean, acc_bad = test(
            opt, test_clean_loader, test_bad_loader, nets, criterions, epoch + 1
        )

        # ========== 第七步：保存模型 ==========
        if opt.save:
            # 判断是否为最佳模型：干净数据准确率是否超过阈值
            is_best = acc_clean[0] > opt.threshold_clean
            # 更新阈值：取后门准确率和当前阈值的较小值
            # 目标是降低ASR（攻击成功率），所以ASR越低越好
            opt.threshold_clean = min(acc_bad[0], opt.threshold_clean)

            # 记录最佳准确率
            best_clean_acc = acc_clean[0]  # 干净数据的Top-1准确率
            best_bad_acc = acc_bad[0]  # 后门数据的Top-1准确率（ASR）

            # 保存模型检查点
            save_checkpoint(
                {
                    'epoch': epoch,  # 当前epoch
                    'state_dict': student.state_dict(),  # 学生网络的参数
                    'best_clean_acc': best_clean_acc,  # 最佳干净准确率
                    'best_bad_acc': best_bad_acc,  # 最佳后门准确率
                    'optimizer': optimizer.state_dict(),  # 优化器状态
                },
                is_best,  # 是否为最佳模型
                opt.checkpoint_root,  # 检查点保存路径
                opt.s_name,  # 学生模型名称
            )


def main():
    """
    程序入口函数

    功能：
    1. 解析命令行参数
    2. 记录训练开始和结束时间
    3. 调用训练函数
    4. 输出总训练时间
    """
    # 解析命令行参数
    opt = get_arguments().parse_args()

    # 记录训练开始时间
    starttime = datetime.datetime.now()

    # 执行训练
    train(opt)

    # 记录训练结束时间
    endtime = datetime.datetime.now()

    # 打印总训练时间
    print(endtime - starttime)


# 程序入口：当直接运行此脚本时执行main函数
if __name__ == '__main__':
    main()
