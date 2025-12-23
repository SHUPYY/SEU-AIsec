"""
argDistillationLoss.py

注意力关系图蒸馏损失（Attention Relation Graph Distillation Loss）

本文件实现了基于注意力关系图（ARG）的知识蒸馏损失函数，
用于后门防御场景中对学生模型进行结构级约束。

核心思想是通过对齐教师模型与学生模型中间层特征所构建的
注意力关系图，引导学生模型学习正常样本下的注意力结构，
从而削弱后门触发器引入的异常注意力关系。

作者：王小康

"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from at import AT
import numpy as np
import matplotlib.pyplot as plt


# =========================
# 基础模块：Linear + BN + ReLU
# =========================
class nn_bn_relu(nn.Module):
    """
    线性映射 + BatchNorm + ReLU
    用于注意力蒸馏中通道特征的维度变换（C → qk_dim）
    """

    def __init__(self, nin, nout):
        """
        :param nin: 输入特征维度（通道数）
        :param nout: 输出特征维度（注意力嵌入维度 qk_dim）
        """
        super(nn_bn_relu, self).__init__()
        self.linear = nn.Linear(nin, nout).cuda()
        self.bn = nn.BatchNorm1d(nout).cuda()
        self.relu = nn.ReLU(False).cuda()

    def forward(self, x, relu=True):
        """
        前向传播
        :param x: 输入特征 [bs, nin]
        :param relu: 是否使用 ReLU 激活
        """
        if relu:
            return self.relu(self.bn(self.linear(x))).cuda()
        return self.bn(self.linear(x)).cuda()


# =========================
# 核心模块：跨层注意力蒸馏
# =========================
class Attention(nn.Module):
    """
    全局跨层注意力蒸馏模块
    用于计算学生各层与教师各层之间的匹配权重，并约束空间注意力一致性
    """

    def __init__(self, args):
        super(Attention, self).__init__()

        # Query / Key 向量维度（论文中固定）
        self.qk_dim = 128

        # 教师层索引映射（用于 unique shape 情况）
        self.n_t = args.n_t

        # 学生与教师的线性变换模块
        self.linear_trans_s = LinearTransformStudent(args)
        self.linear_trans_t = LinearTransformTeacher(args)

        # 可学习的层嵌入参数（建模层间先验关系）
        self.p_t = nn.Parameter(torch.Tensor(len(args.t_shapes), self.qk_dim)).cuda()
        self.p_s = nn.Parameter(torch.Tensor(len(args.s_shapes), self.qk_dim)).cuda()

        # Xavier 初始化
        torch.nn.init.xavier_normal_(self.p_t)
        torch.nn.init.xavier_normal_(self.p_s)

    def forward(self, g_s, g_t):
        """
        :param g_s: 学生模型多层特征图列表
        :param g_t: 教师模型多层特征图列表
        :return: 每一层教师对应的蒸馏损失
        """

        # 学生侧：计算 bilinear key 和空间注意力
        bilinear_key, h_hat_s_all = self.linear_trans_s(g_s)

        # 教师侧：计算 query 和空间注意力
        query, h_t_all = self.linear_trans_t(g_t)

        # 层嵌入的内积，用于提供层间先验
        p_logit = torch.matmul(self.p_t, self.p_s.t())

        # 注意力匹配得分（scaled dot-product attention）
        logit = (
            torch.einsum('bstq,btq->bts', bilinear_key, query) + p_logit
        ) / np.sqrt(self.qk_dim)

        # 归一化得到跨层注意力权重
        atts = F.softmax(logit, dim=2)

        # 针对每一层教师特征计算加权蒸馏损失
        loss = []
        for i, (n, h_t) in enumerate(zip(self.n_t, h_t_all)):
            h_hat_s = h_hat_s_all[n]
            diff = self.cal_diff(h_hat_s, h_t, atts[:, i])
            loss.append(diff)

        return loss

    def cal_diff(self, v_s, v_t, att):
        """
        计算学生与教师空间注意力之间的加权 MSE 损失
        :param v_s: 学生空间注意力 [bs, s_layers, HW]
        :param v_t: 教师空间注意力 [bs, HW]
        :param att: 学生-教师层匹配权重
        """
        diff = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
        diff = torch.mul(diff, att).sum(1).mean()
        return diff


# =========================
# 教师侧特征线性变换
# =========================
class LinearTransformTeacher(nn.Module):
    """
    教师模型特征 → Query / Value
    """

    def __init__(self, args):
        super(LinearTransformTeacher, self).__init__()
        self.qk_dim = 128

        # 每一层教师特征使用独立的线性映射
        self.query_layer = nn.ModuleList(
            [nn_bn_relu(t_shape[0], self.qk_dim) for t_shape in args.t_shapes]
        )

    def forward(self, g_t):
        """
        :param g_t: 教师模型多层特征图
        :return: query（通道注意力），value（空间注意力）
        """
        bs = g_t[0].size(0)

        # 通道注意力：对空间维度求均值
        channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]

        # 空间注意力：对通道平方求均值
        spatial_mean = [f_t.pow(2).mean(1).view(bs, -1) for f_t in g_t]

        # 生成 query
        query = torch.stack(
            [
                query_layer(f_t, relu=False)
                for f_t, query_layer in zip(channel_mean, self.query_layer)
            ],
            dim=1,
        )

        # 归一化空间注意力
        value = [F.normalize(f_s, dim=1) for f_s in spatial_mean]

        return query, value


# =========================
# 学生侧特征线性变换
# =========================
class LinearTransformStudent(nn.Module):
    """
    学生模型特征 → Key / Value，并构造跨层双线性映射
    """

    def __init__(self, args):
        super(LinearTransformStudent, self).__init__()
        self.t = len(args.t_shapes)
        self.s = len(args.s_shapes)
        self.qk_dim = 128

        # 将学生特征采样到教师空间尺寸
        self.samplers = nn.ModuleList(
            [Sample(t_shape) for t_shape in args.unique_t_shapes]
        )

        # Key 映射层
        self.key_layer = nn.ModuleList(
            [nn_bn_relu(s_shape[0], self.qk_dim) for s_shape in args.s_shapes]
        ).cuda()

        # 双线性映射，用于构造跨层关系
        self.bilinear = nn_bn_relu(self.qk_dim, self.qk_dim * self.t)

    def forward(self, g_s):
        """
        :param g_s: 学生模型多层特征图
        :return: bilinear_key（跨层 key），空间注意力 value
        """
        bs = g_s[0].size(0)

        # 通道注意力
        channel_mean = [f_s.mean(3).mean(2).cuda() for f_s in g_s]

        # 空间注意力（采样到教师尺寸）
        spatial_mean = [sampler(g_s, bs) for sampler in self.samplers]

        # 生成 key
        key = torch.stack(
            [
                key_layer(f_s.cuda())
                for key_layer, f_s in zip(self.key_layer, channel_mean)
            ],
            dim=1,
        ).view(bs * self.s, -1)

        # 构造双线性 key
        bilinear_key = self.bilinear(key, relu=False).view(bs, self.s, self.t, -1)

        # 归一化空间注意力
        value = [F.normalize(s_m, dim=2) for s_m in spatial_mean]

        return bilinear_key, value


# =========================
# 特征图空间尺寸采样模块
# =========================
class Sample(nn.Module):
    """
    使用 AdaptiveAvgPool2d
    将学生特征图采样到教师特征图的空间尺寸
    """

    def __init__(self, t_shape):
        super(Sample, self).__init__()
        _, t_H, t_W = t_shape
        self.sample = nn.AdaptiveAvgPool2d((t_H, t_W))

    def forward(self, g_s, bs):
        """
        :param g_s: 学生多层特征图
        :param bs: batch size
        :return: 采样后的空间注意力
        """
        g_s = torch.stack(
            [self.sample(f_s.pow(2).mean(1, keepdim=True)).view(bs, -1) for f_s in g_s],
            dim=1,
        )
        return g_s


# =========================
# 损失函数主体
# =========================
class argDL(nn.Module):
    """
    Attention-based Relational Graph Distillation
    用于后门防御的蒸馏损失
    """

    def __init__(self, opt):
        super(argDL, self).__init__()
        self.w_argDL_vert = opt.w_argDL_vert  # 节点注意力权重
        self.w_argDL_edge = opt.w_argDL_edge  # 边关系权重
        self.attention = Attention(opt)

    def forward(self, irg_s, irg_t):
        """
        :param irg_s: 学生模型中间层特征
        :param irg_t: 教师模型中间层特征
        """
        fm_s0, fm_s1, fm_s2 = irg_s
        fm_t0, fm_t1, fm_t2 = irg_t

        # 节点级注意力蒸馏（AT）
        criterionAT = AT(2.0)
        loss_argDL_vert = (
            criterionAT(fm_s0, fm_t0) * 2000
            + criterionAT(fm_s1, fm_t1) * 5000
            + criterionAT(fm_s2, fm_t2) * 2000
        )

        # 计算注意力图
        fm_s0_attention = criterionAT.attention_map(fm_s0)
        fm_t0_attention = criterionAT.attention_map(fm_t0)
        fm_s1_attention = criterionAT.attention_map(fm_s1)
        fm_t1_attention = criterionAT.attention_map(fm_t1)
        fm_s2_attention = criterionAT.attention_map(fm_s2)
        fm_t2_attention = criterionAT.attention_map(fm_t2)

        # 全局跨层注意力蒸馏
        loss_global = sum(
            self.attention(
                [fm_s0, fm_s1, fm_s2],
                [fm_t0, fm_t1, fm_t2],
            )
        )

        # 层间关系蒸馏（边）
        loss_argDL_edge = (
            F.mse_loss(
                self.euclidean_dist_fms(fm_s0_attention, fm_s1_attention, True),
                self.euclidean_dist_fms(fm_t0_attention, fm_t1_attention, True),
            )
            + F.mse_loss(
                self.euclidean_dist_fms(fm_s1_attention, fm_s2_attention, True),
                self.euclidean_dist_fms(fm_t1_attention, fm_t2_attention, True),
            )
            + F.mse_loss(
                self.euclidean_dist_fms(fm_s0_attention, fm_s2_attention, True),
                self.euclidean_dist_fms(fm_t0_attention, fm_t2_attention, True),
            )
        ) / 3

        # 总损失
        loss = (
            self.w_argDL_vert * loss_argDL_vert
            + self.w_argDL_edge * loss_argDL_edge
            + loss_global
        )

        return loss
