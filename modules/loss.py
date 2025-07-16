from torch import nn
from torch import Tensor
import torch
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        return ((x_hat - x)**2).sum(axis=-1)


class CategoricalReconstuctionLoss(nn.Module):
    def __init__(self, n_cat_feats: int) -> None:
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss()
        self.n_cat_feats = n_cat_feats

    def forward(self, x_hat: Tensor, x: Tensor) -> Tensor:
        reconstr = self.reconstruction_loss(
            x_hat[:, :-self.n_cat_feats],
            x[:, :-self.n_cat_feats]
        )
        if self.n_cat_feats > 0:
            cat_reconstr = nn.functional.binary_cross_entropy_with_logits(
                x_hat[:, -self.n_cat_feats:],
                x[:, -self.n_cat_feats:],
                reduction='none'
            ).sum(axis=-1)
            reconstr += cat_reconstr
        return reconstr


class QuantizeLoss(nn.Module):
    def __init__(self, commitment_weight: float = 1.0) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight

    def forward(self, query: Tensor, value: Tensor) -> Tensor:
        emb_loss = ((query.detach() - value)**2).sum(axis=[-1])
        query_loss = ((query - value.detach())**2).sum(axis=[-1])
        return emb_loss + self.commitment_weight * query_loss


# 修改后的标签对齐损失函数，使用InfoNCE对比学习损失
class TagAlignmentLoss(nn.Module):
    def __init__(self, alignment_weight: float = 1.0, temperature: float = 0.1) -> None:
        super().__init__()
        self.alignment_weight = alignment_weight
        self.temperature = temperature

    def forward(self, codebook_emb: Tensor, tag_emb: Tensor, layer_idx: int) -> Tensor:
        """
        计算码本嵌入和标签嵌入之间的对比学习损失 (InfoNCE)

        参数:
            codebook_emb: 形状为 [batch_size, embed_dim] 的码本嵌入
            tag_emb: 形状为 [batch_size, embed_dim] 的标签嵌入
            layer_idx: 当前层的索引，用于调整不同层的对齐策略
        """
        batch_size = codebook_emb.size(0)

        # 归一化嵌入向量
        codebook_emb_norm = F.normalize(codebook_emb, p=2, dim=-1)
        tag_emb_norm = F.normalize(tag_emb, p=2, dim=-1)

        # 计算点积
        logits = torch.matmul(codebook_emb_norm, tag_emb_norm.transpose(0, 1)) / self.temperature

        # 对角线位置是正样本
        positive_logits = torch.diag(logits)

        # InfoNCE损失计算
        labels = torch.arange(batch_size, device=codebook_emb.device)
        loss = F.cross_entropy(logits, labels)

        # 根据层索引调整权重
        layer_weight = 1.0 / ((layer_idx * 0.5) + 1)  # 修改：调整层级权重策略

        # 总损失 = InfoNCE损失。之前包含一个可能为负的项，现已移除。
        total_loss = loss * self.alignment_weight * layer_weight

        return total_loss


# 改进标签预测损失函数，添加更灵活的焦点损失参数调整
class TagPredictionLoss(nn.Module):
    def __init__(self, use_focal_loss: bool = False, focal_params: dict = None, class_counts: dict = None) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.focal_params = focal_params or {'gamma': 2.0, 'alpha': 0.25}
        self.class_counts = class_counts  # 各层各类别的频率统计

        # 添加标签平滑参数
        self.use_label_smoothing = True
        self.label_smoothing_alpha = 0.1

        # 用于缓解过拟合的额外策略
        self.use_mixup = True
        self.mixup_alpha = 0.2  # 控制混合程度

        # 学习率衰减的权重调整器
        self.weight_scheduler = None

    def forward(self, pred_logits: Tensor, target_indices: Tensor, layer_idx: int = 0) -> tuple:
        """
        计算标签预测损失和准确率，支持焦点损失

        参数:
            pred_logits: 形状为 [batch_size, num_classes] 的预测logits
            target_indices: 形状为 [batch_size] 的目标标签索引
            layer_idx: 当前层的索引，用于获取该层的焦点损失参数

        返回:
            loss: 交叉熵损失或焦点损失
            accuracy: 预测准确率
        """
        # 确保目标索引是有效的（不是-1）
        valid_mask = (target_indices >= 0)

        if valid_mask.sum() == 0:
            # 如果没有有效的目标，返回零损失和准确率
            return torch.tensor(0.0, device=pred_logits.device), torch.tensor(0.0, device=pred_logits.device)

        # 只计算有效目标的损失
        valid_logits = pred_logits[valid_mask]
        valid_targets = target_indices[valid_mask]

        # 计算准确率
        pred_indices = torch.argmax(valid_logits, dim=-1)
        accuracy = (pred_indices == valid_targets).float().mean()

        # 获取模型预测的概率分布
        probs = F.softmax(valid_logits, dim=-1)

        # 应用混合策略 (Mixup) - 随机混合样本来增强泛化能力
        if self.use_mixup and valid_logits.size(0) > 1:
            # 只在训练阶段使用mixup
            if valid_logits.requires_grad:
                batch_size = valid_logits.size(0)
                # 创建随机排列的索引
                indices = torch.randperm(batch_size, device=valid_logits.device)
                # 生成混合权重
                lam = torch.distributions.Beta(torch.tensor(self.mixup_alpha),
                                              torch.tensor(self.mixup_alpha)).sample().to(valid_logits.device)

                # 混合logits和目标
                mixed_logits = lam * valid_logits + (1 - lam) * valid_logits[indices]
                valid_logits = mixed_logits

                # 保存原始和混合的目标，用于计算损失
                targets_a, targets_b = valid_targets, valid_targets[indices]

        # 根据是否使用焦点损失选择不同的损失计算方式
        if self.use_focal_loss:
            # 获取当前层的焦点损失参数，对深层使用更高的gamma值
            gamma = self.focal_params.get(f'gamma_{layer_idx}', self.focal_params.get('gamma', 2.0)) * (1 + 0.35 * layer_idx)

            # 较小的类别平衡因子，避免过度补偿
            alpha = max(0.08, self.focal_params.get(f'alpha_{layer_idx}', self.focal_params.get('alpha', 0.25)) - 0.06 * layer_idx)

            # 如果有类别频率统计，则使用动态权重
            if self.class_counts is not None and layer_idx in self.class_counts:
                class_counts = self.class_counts[layer_idx]
                if isinstance(class_counts, torch.Tensor) and class_counts.numel() > 0:
                    # 计算类别权重：频率越低，权重越高
                    class_freq = class_counts.float() / class_counts.sum()
                    # 防止除零错误
                    class_freq = torch.clamp(class_freq, min=1e-6)
                    # 权重与频率成反比，更强调稀有类别，但避免极端权重
                    weights = 1.0 / torch.sqrt(class_freq)
                    # 归一化权重，但减小极值
                    weights = torch.clamp(weights / weights.mean(), min=0.5, max=3.0)
                    # 确保权重张量的设备与logits一致
                    weights = weights.to(valid_logits.device)

                    # 使用类别权重计算焦点损失，同时应用标签平滑
                    if self.use_mixup and valid_logits.requires_grad and valid_logits.size(0) > 1:
                        # 如果使用了mixup，对两组目标分别计算损失
                        loss_a = self._focal_loss_with_weights_and_smoothing(valid_logits, targets_a, gamma, weights)
                        loss_b = self._focal_loss_with_weights_and_smoothing(valid_logits, targets_b, gamma, weights)
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        loss = self._focal_loss_with_weights_and_smoothing(valid_logits, valid_targets, gamma, weights)
                else:
                    # 没有有效的类别统计，使用标准焦点损失
                    if self.use_mixup and valid_logits.requires_grad and valid_logits.size(0) > 1:
                        loss_a = self._focal_loss_with_smoothing(valid_logits, targets_a, gamma, alpha)
                        loss_b = self._focal_loss_with_smoothing(valid_logits, targets_b, gamma, alpha)
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        loss = self._focal_loss_with_smoothing(valid_logits, valid_targets, gamma, alpha)
            else:
                # 使用标准焦点损失
                if self.use_mixup and valid_logits.requires_grad and valid_logits.size(0) > 1:
                    loss_a = self._focal_loss_with_smoothing(valid_logits, targets_a, gamma, alpha)
                    loss_b = self._focal_loss_with_smoothing(valid_logits, targets_b, gamma, alpha)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss = self._focal_loss_with_smoothing(valid_logits, valid_targets, gamma, alpha)
        else:
            # 使用标准交叉熵损失，但添加标签平滑和正则化
            label_smoothing = min(0.25, 0.05 + layer_idx * 0.06)  # 深层使用更多平滑

            # 应用L2正则化
            weight_decay = 0.01 * (1 + layer_idx * 0.5)  # 随层数增加而增大
            l2_reg = torch.tensor(0.0, device=valid_logits.device)
            for param in pred_logits.parameters() if hasattr(pred_logits, 'parameters') else []:
                l2_reg += torch.norm(param, 2)

            # 使用交叉熵损失并应用标签平滑
            if self.use_mixup and valid_logits.requires_grad and valid_logits.size(0) > 1:
                loss_a = F.cross_entropy(valid_logits, targets_a, reduction='mean', label_smoothing=label_smoothing)
                loss_b = F.cross_entropy(valid_logits, targets_b, reduction='mean', label_smoothing=label_smoothing)
                ce_loss = lam * loss_a + (1 - lam) * loss_b
            else:
                ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction='mean', label_smoothing=label_smoothing)

            # 额外添加一个KL散度正则项，防止过拟合
            uniform = torch.ones_like(probs) / probs.size(-1)
            kl_div = F.kl_div(torch.log(probs + 1e-8), uniform, reduction='batchmean') * 0.05

            # 合并损失
            loss = ce_loss + weight_decay * l2_reg + kl_div

        return loss, accuracy

    def _focal_loss_with_smoothing(self, logits: Tensor, targets: Tensor, gamma: float = 2.0, alpha: float = 0.25) -> Tensor:
        """
        带标签平滑的焦点损失

        参数:
            logits: 预测的logits
            targets: 目标类别索引
            gamma: 聚焦参数，减少易分类样本的权重
            alpha: 平衡参数，处理类别不平衡
        """
        num_classes = logits.size(-1)
        batch_size = targets.size(0)

        # 创建one-hot编码
        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)

        # 应用标签平滑
        if self.use_label_smoothing and logits.requires_grad:
            # 根据类别数量调整平滑度 - 类别越多，平滑度越高
            class_factor = min(0.3, 0.05 * (num_classes / 100))  # 类别数量影响因子
            smoothing = min(0.25, self.label_smoothing_alpha + gamma * 0.015 + class_factor) # 调整 gamma 影响因子
            one_hot = one_hot * (1 - smoothing) + smoothing / num_classes

        # 计算焦点损失
        probs = F.softmax(logits, dim=-1)
        pt = (one_hot * probs).sum(dim=1)
        focal_weight = alpha * ((1 - pt) ** gamma)

        # 计算交叉熵损失
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(one_hot * log_probs, dim=1)

        # 应用焦点权重
        focal_loss = focal_weight * loss

        return focal_loss.mean()

    def _focal_loss_with_weights_and_smoothing(self, logits: Tensor, targets: Tensor, gamma: float = 2.0, class_weights: Tensor = None) -> Tensor:
        """
        带标签平滑和类别权重的焦点损失

        参数:
            logits: 预测的logits
            targets: 目标类别索引
            gamma: 聚焦参数，减少易分类样本的权重
            class_weights: 每个类别的权重
        """
        num_classes = logits.size(-1)
        batch_size = targets.size(0)

        # 创建one-hot编码
        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)

        # 应用标签平滑 - 针对大类别数的层使用更强的平滑
        if self.use_label_smoothing and logits.requires_grad:
            # 根据类别数量调整平滑度 - 类别越多，平滑度越高
            class_factor = min(0.3, 0.05 * (num_classes / 100))  # 类别数量影响因子
            smoothing = min(0.25, self.label_smoothing_alpha + gamma * 0.015 + class_factor) # 调整 gamma 影响因子
            one_hot = one_hot * (1 - smoothing) + smoothing / num_classes

        # 获取每个样本对应类别的权重
        sample_weights = class_weights[targets] if class_weights is not None else torch.ones_like(targets, dtype=torch.float)

        # 计算焦点损失
        probs = F.softmax(logits, dim=-1)
        pt = (one_hot * probs).sum(dim=1)

        # 针对类别数量多的情况，调整gamma值
        # 类别数量越多，gamma值越大，更关注难分类样本
        adjusted_gamma = gamma * (1.0 + 0.25 * min(1.0, num_classes / 250)) # 调整 num_classes 影响因子
        focal_weight = ((1 - pt) ** adjusted_gamma)

        # 应用样本权重和焦点权重
        weighted_focal = sample_weights * focal_weight

        # 计算交叉熵损失
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(one_hot * log_probs, dim=1)

        # 应用加权焦点损失
        focal_loss = weighted_focal * loss

        # 添加额外的正则化项，防止过拟合
        # 对于类别数量多的层，添加更强的正则化
        if num_classes > 100 and logits.requires_grad:
            # 鼓励预测分布更加均匀，避免过度自信
            uniform = torch.ones_like(probs) / num_classes
            kl_div = F.kl_div(torch.log(probs + 1e-8), uniform, reduction='batchmean')
            reg_weight = min(0.12, 0.015 * (num_classes / 100))  # 根据类别数量调整权重
            return focal_loss.mean() + reg_weight * kl_div

        return focal_loss.mean()