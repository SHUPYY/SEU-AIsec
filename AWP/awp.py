# import copy
# import torch
# import torch.nn as nn
# import torch.optim as optim

# EPS = 1E-20


# def normalize(perturbations, weights):
#     """
#     将扰动梯度的范数归一化到与原始权重相同的尺度
#     """
#     perturbations.mul_(weights.norm() / (perturbations.norm() + EPS))


# # ============================================================
# # 原始 AWP 梯度归一化函数（未分层）
# # 【保留原始实现，仅注释，不删除】
# # ============================================================
# """
# def normalize_grad_by_weights(weights, ref_weights):
#     for w, ref_w in zip(weights, ref_weights):
#         if w.dim() <= 1:
#             # 忽略 BN 参数和 bias（1维或0维）
#             w.grad.data.fill_(0)
#         else:
#             # 对梯度按权重范数进行归一化
#             normalize(w.grad.data, ref_w)
# """


# # ============================================================
# # 【MOD】分层 AWP 梯度归一化（Layer-wise AWP）
# # ============================================================
# def normalize_grad_by_weights(model, ref_weights):
#     """
#     对不同层施加不同强度的权重扰动（Layer-wise AWP）

#     参数：
#         model       : 当前模型（用于获取参数名）
#         ref_weights : 原始权重（扰动前）
#     """
#     for (name, w), ref_w in zip(model.named_parameters(), ref_weights):

#         # 1️⃣ 忽略 BN 和 bias（AWP 原论文做法）
#         if w.dim() <= 1:
#             w.grad.data.fill_(0)
#             continue

#         # 2️⃣ 分层扰动系数（核心改进点）
#         #    后层 > 前层
#         if 'fc' in name or 'linear' in name:
#             coef = 1.0        # 分类头，扰动最大
#         elif 'layer3' in name or 'layer4' in name:
#             coef = 0.75       # 高层特征
#         else:
#             coef = 0.5        # 低层特征，扰动较小

#         # 3️⃣ 梯度按权重尺度归一化
#         normalize(w.grad.data, ref_w)

#         # 4️⃣ 施加分层系数
#         w.grad.data.mul_(coef)


# class AdvWeightPerturb(object):
#     """
#     AWP（Adversarial Weight Perturbation）
#     本实现用于 标准对抗训练（PGD-AT / TRADES-AT 均可接入）
#     """

#     def __init__(self, model, eta, nb_iter=1):
#         """
#         参数：
#             model   : 被扰动的模型
#             eta     : AWP 扰动强度（γ）
#             nb_iter : 权重扰动的迭代步数（通常=1）
#         """
#         super(AdvWeightPerturb, self).__init__()

#         self.model = model
#         self.eta = eta
#         self.nb_iter = nb_iter

#         # 使用 SGD 在“权重空间”中做梯度上升
#         self.optim = optim.SGD(model.parameters(), lr=eta / nb_iter)

#         self.criterion = nn.CrossEntropyLoss()
#         self.diff = None  # 保存最终权重扰动

#     def perturb(self, X_adv, y):
#         """
#         在权重空间执行对抗扰动
#         """
#         # ====================================================
#         # 1️⃣ 保存原始权重（深拷贝）
#         # ====================================================
#         old_w = copy.deepcopy([p.data for p in self.model.parameters()])

#         # ====================================================
#         # 2️⃣ 在权重空间进行对抗攻击（最大化 loss）
#         # ====================================================
#         for _ in range(self.nb_iter):
#             self.optim.zero_grad()

#             outputs = self.model(X_adv)

#             # 负号：最大化损失（对抗）
#             loss = -self.criterion(outputs, y)
#             loss.backward()

#             # ====================================================
#             # 原始 AWP 调用（未分层）
#             # 【保留但注释】
#             # ====================================================
#             # normalize_grad_by_weights(self.model.parameters(), old_w)

#             # ====================================================
#             # 【MOD】分层 AWP 梯度归一化
#             # ====================================================
#             normalize_grad_by_weights(self.model, old_w)

#             # 执行一次权重更新（扰动）
#             self.optim.step()

#         # ====================================================
#         # 3️⃣ 记录最终权重扰动 ΔW
#         # ====================================================
#         self.diff = [
#             w.data - w_old for w, w_old in zip(self.model.parameters(), old_w)
#         ]

#     def restore(self):
#         """
#         恢复原始权重（W = W_adv - ΔW）
#         """
#         for w, v in zip(self.model.parameters(), self.diff):
#             w.data.sub_(v.data)
import copy
import torch
import torch.nn as nn
import torch.optim as optim

EPS = 1e-20


def normalize(perturbations, weights):
    """
    将扰动梯度的范数归一化到与原始权重相同的尺度
    """
    perturbations.mul_(weights.norm() / (perturbations.norm() + EPS))


# ============================================================
# 原始 AWP 梯度归一化函数（未分层）
# 【保留原始实现，仅注释，不删除】
# ============================================================
"""
def normalize_grad_by_weights(weights, ref_weights):
    for w, ref_w in zip(weights, ref_weights):
        if w.dim() <= 1:
            # 忽略 BN 参数和 bias（1维或0维）
            w.grad.data.fill_(0)
        else:
            # 对梯度按权重范数进行归一化
            normalize(w.grad.data, ref_w)
"""


# ============================================================
# 【MOD】分层 AWP 梯度归一化（Layer-wise AWP）
# ============================================================
def normalize_grad_by_weights(model, ref_weights):
    """
    对不同层施加不同强度的权重扰动（Layer-wise AWP）

    参数：
        model       : 当前模型（用于获取参数名）
        ref_weights : 原始权重（扰动前）
    """
    for (name, w), ref_w in zip(model.named_parameters(), ref_weights):

        # 1️⃣ 忽略 BN 和 bias（AWP 原论文做法）
        if w.dim() <= 1:
            w.grad.data.fill_(0)
            continue

        # 2️⃣ 分层扰动系数（核心改进点）
        #    后层 > 前层
        if 'fc' in name or 'linear' in name:
            coef = 1.0  # 分类头，扰动最大
        elif 'layer3' in name or 'layer4' in name:
            coef = 0.75  # 高层特征
        else:
            coef = 0.5  # 低层特征，扰动较小

        # 3️⃣ 梯度按权重尺度归一化
        normalize(w.grad.data, ref_w)

        # 4️⃣ 施加分层系数
        w.grad.data.mul_(coef)


class AdvWeightPerturb(object):
    """
    AWP（Adversarial Weight Perturbation）
    本实现用于 标准对抗训练（PGD-AT / TRADES-AT 均可接入）
    """

    def __init__(self, model, eta, nb_iter=1):
        """
        参数：
            model   : 被扰动的模型
            eta     : AWP 扰动强度（γ）
            nb_iter : 权重扰动的迭代步数（通常=1）
        """
        super(AdvWeightPerturb, self).__init__()

        self.model = model
        self.eta = eta
        self.nb_iter = nb_iter

        # 使用 SGD 在“权重空间”中做梯度上升
        self.optim = optim.SGD(model.parameters(), lr=eta / nb_iter)

        self.criterion = nn.CrossEntropyLoss()
        self.diff = None  # 保存最终权重扰动

    def perturb(self, X_adv, y):
        """
        在权重空间执行对抗扰动
        """
        # ====================================================
        # 1️⃣ 保存原始权重（深拷贝）
        # ====================================================
        old_w = copy.deepcopy([p.data for p in self.model.parameters()])

        # ====================================================
        # 2️⃣ 在权重空间进行对抗攻击（最大化 loss）
        # ====================================================
        for _ in range(self.nb_iter):
            self.optim.zero_grad()

            outputs = self.model(X_adv)

            # 负号：最大化损失（对抗）
            loss = -self.criterion(outputs, y)
            loss.backward()

            # ====================================================
            # 原始 AWP 调用（未分层）
            # 【保留但注释】
            # ====================================================
            # normalize_grad_by_weights(self.model.parameters(), old_w)

            # ====================================================
            # 【MOD】分层 AWP 梯度归一化
            # ====================================================
            normalize_grad_by_weights(self.model, old_w)

            # 执行一次权重更新（扰动）
            self.optim.step()

        # ====================================================
        # 3️⃣ 记录最终权重扰动 ΔW
        # ====================================================
        self.diff = [w.data - w_old for w, w_old in zip(self.model.parameters(), old_w)]

    def restore(self):
        """
        恢复原始权重（W = W_adv - ΔW）
        """
        for w, v in zip(self.model.parameters(), self.diff):
            w.data.sub_(v.data)


# import copy
# import torch
# import torch.nn as nn
# import torch.optim as optim

# EPS = 1e-20


# def normalize(perturb, weight):
#     """
#     对扰动进行归一化，使其范数与参考权重一致
#     """
#     perturb.mul_(weight.norm() / (perturb.norm() + EPS))


# # =========================
# # 原始 AWP
# # =========================
# # def normalize_grad_by_weights(weights, ref_weights):
# #     for w, ref_w in zip(weights, ref_weights):
# #         if w.dim() <= 1:
# #             w.grad.data.fill_(0)
# #         else:
# #             normalize(w.grad.data, ref_w)


# # =====================================================
# # ★ Layer-wise AWP v1
# # =====================================================
# def normalize_grad_by_weights(weights, ref_weights):
#     """
#     Layer-wise AWP 核心改动：
#     - 每一层单独进行扰动归一化
#     - BN / bias 层不施加扰动
#     """
#     for w, ref_w in zip(weights, ref_weights):
#         if w.dim() <= 1:
#             # BN / bias 不参与权重扰动
#             w.grad.data.zero_()
#         else:
#             # 按层归一化扰动
#             normalize(w.grad.data, ref_w)


# class AdvWeightPerturb(object):
#     """
#     Layer-wise AWP（PGD-5 = 57.13）
#     """

#     def __init__(self, model, eta, nb_iter=1):
#         self.model = model
#         self.eta = eta
#         self.nb_iter = nb_iter

#         self.optimizer = optim.SGD(self.model.parameters(), lr=eta / nb_iter)
#         self.criterion = nn.CrossEntropyLoss()
#         self.diff = None

#     def perturb(self, X_adv, y):
#         """
#         在权重空间中执行对抗扰动
#         """
#         # 1. 保存原始权重
#         old_weights = copy.deepcopy([p.data for p in self.model.parameters()])

#         # 2. 对抗扰动优化
#         for _ in range(self.nb_iter):
#             self.optimizer.zero_grad()
#             output = self.model(X_adv)
#             loss = -self.criterion(output, y)
#             loss.backward()

#             # ★ Layer-wise 归一化
#             normalize_grad_by_weights(self.model.parameters(), old_weights)

#             self.optimizer.step()

#         # 3. 保存权重扰动
#         self.diff = [
#             w.data - old_w for w, old_w in zip(self.model.parameters(), old_weights)
#         ]

#     def restore(self):
#         """
#         恢复权重
#         """
#         for w, d in zip(self.model.parameters(), self.diff):
#             w.data.sub_(d)
