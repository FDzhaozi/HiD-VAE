import torch
import torch.nn.functional as F

from data.schemas import SeqBatch, HRqVaeOutput, HRqVaeComputedLosses
from einops import rearrange
from functools import cached_property
from modules.encoder import MLP
from modules.loss import CategoricalReconstuctionLoss
from modules.loss import ReconstructionLoss
from modules.loss import QuantizeLoss
from modules.loss import TagAlignmentLoss
from modules.loss import TagPredictionLoss
from modules.normalize import l2norm
from modules.quantize import Quantize
from modules.quantize import QuantizeForwardMode
from huggingface_hub import PyTorchModelHubMixin
from typing import List, Dict, Optional
from torch import nn
from torch import Tensor

torch.set_float32_matmul_precision('high')


# 添加一个新的损失函数类，用于计算语义ID唯一性约束损失
class SemanticIdUniquenessLoss(nn.Module):
    """
    计算语义ID唯一性约束损失，推动不同商品的语义ID彼此分离
    """
    def __init__(self, margin: float = 0.5, weight: float = 1.0):
        """
        初始化语义ID唯一性约束损失
        
        参数:
            margin: 语义ID之间的最小距离阈值
            weight: 损失权重
        """
        super().__init__()
        self.margin = margin
        self.weight = weight
    
    def forward(self, sem_ids: Tensor, encoded_features: Tensor) -> Tensor:
        """
        计算批次内语义ID的唯一性损失
        
        参数:
            sem_ids: 形状为 [batch_size, n_layers] 的语义ID
            encoded_features: 形状为 [batch_size, embed_dim] 的编码器输出
            
        返回:
            唯一性约束损失
        """
        batch_size, n_layers = sem_ids.shape
        
        # 如果批次大小太小，不计算损失
        if batch_size <= 1:
            return torch.tensor(0.0, device=sem_ids.device)
        
        # 找到具有完全相同语义ID的对
        # 展开为 [batch_size, 1, n_layers] 和 [1, batch_size, n_layers]
        id1 = sem_ids.unsqueeze(1)
        id2 = sem_ids.unsqueeze(0)
        
        # 检查所有层是否都相等
        id_eq = (id1 == id2).all(dim=-1)
        
        # 创建对角线掩码，排除自身比较
        diag_mask = ~torch.eye(batch_size, device=sem_ids.device, dtype=torch.bool)
        
        # 找出完全相同的ID对 (非自身)
        identical_pairs_mask = id_eq & diag_mask
        
        # 如果没有完全相同的ID对，返回零损失
        if not identical_pairs_mask.any():
            return torch.tensor(0.0, device=sem_ids.device)
        
        # 获取相同ID对的索引
        idx_a, idx_b = torch.where(identical_pairs_mask)
        
        # 为避免重复计算，只考虑 i < j 的对
        unique_pairs_mask = idx_a < idx_b
        idx_a = idx_a[unique_pairs_mask]
        idx_b = idx_b[unique_pairs_mask]

        if len(idx_a) == 0:
            return torch.tensor(0.0, device=sem_ids.device)
            
        # 获取这些对的编码特征
        features_a = encoded_features[idx_a]
        features_b = encoded_features[idx_b]
        
        # 归一化特征以计算余弦相似度
        features_a_norm = F.normalize(features_a, p=2, dim=-1)
        features_b_norm = F.normalize(features_b, p=2, dim=-1)

        # 计算余弦相似度
        cosine_sim = (features_a_norm * features_b_norm).sum(dim=-1)
        
        # 计算损失：我们希望将这些特征推开，因此当相似度高于margin时，我们施加一个惩罚
        # 损失 = max(0, cosine_sim - margin)
        loss = F.relu(cosine_sim - self.margin)

        # 对所有冲突对的损失求平均，并乘以权重
        uniqueness_loss = self.weight * loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=sem_ids.device)

        return uniqueness_loss


class TagPredictor(nn.Module):
    """
    标签预测器，用于预测每层对应的标签索引
    """
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True,
        layer_idx: int = 0
    ) -> None:
        super().__init__()
        
        # 如果未指定隐藏层维度，则使用嵌入维度的2倍
        if hidden_dim is None:
            hidden_dim = embed_dim * 2
        
        # 根据层索引调整dropout率，越深的层dropout率越高
        # 但不要太高，避免信息丢失过多
        dropout_rate = min(0.55, dropout_rate + layer_idx * 0.075)
        
        # 自注意力机制 - 帮助模型关注输入特征中的重要部分
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Sigmoid()
        )
        
        # 构建更深、更宽的分类器网络，包含残差连接
        # 第一部分：特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 中间层尺寸
        mid_dim = int(hidden_dim * 0.9)
        
        # 第二部分：残差模块
        self.residual_block1 = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mid_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim) if use_batch_norm else nn.Identity(),
        )
        
        # 第三部分：残差模块2
        self.residual_block2 = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mid_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim) if use_batch_norm else nn.Identity(),
        )
        
        # 输出层
        classifier_mid_dim = mid_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_mid_dim),
            nn.LayerNorm(classifier_mid_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_mid_dim, classifier_mid_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(classifier_mid_dim // 2, num_classes)
        )
        
        # 标签平滑正则化参数
        self.label_smoothing = 0.1 if layer_idx > 0 else 0.05
        
        # 特征归一化
        self.apply_norm = layer_idx > 0  # 对深层使用归一化
    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        参数:
            x: 形状为 [batch_size, embed_dim] 的输入嵌入
            
        返回:
            logits: 形状为 [batch_size, num_classes] 的预测logits
        """
        # 应用自注意力机制
        attention_weights = self.attention(x)
        x_attended = x * attention_weights
        
        # 特征归一化（可选）
        if self.apply_norm:
            x_attended = F.normalize(x_attended, p=2, dim=-1)
        
        # 特征提取
        features = self.feature_extractor(x_attended)
        
        # 应用残差模块
        res1 = self.residual_block1(features)
        features = features + res1
        
        res2 = self.residual_block2(features)
        features = features + res2
        
        # 分类
        logits = self.classifier(features)
        
        return logits


class HRqVae(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        codebook_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        codebook_sim_vq: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        n_layers: int = 3,
        commitment_weight: float = 0.25,
        n_cat_features: int = 18,
        tag_alignment_weight: float = 0.5,
        tag_prediction_weight: float = 0.5,
        tag_class_counts: Optional[List[int]] = None,
        tag_embed_dim: int = 768,  # 标签嵌入维度
        use_focal_loss: bool = False,  # 是否使用焦点损失
        focal_loss_params: Optional[Dict] = None,  # 焦点损失参数
        dropout_rate: float = 0.2,  # 新增：Dropout率
        use_batch_norm: bool = True,  # 新增：是否使用BatchNorm
        alignment_temperature: float = 0.1,  # 新增：对比学习温度参数
        # 新增：语义ID唯一性约束参数
        sem_id_uniqueness_weight: float = 0.5,  # 语义ID唯一性约束权重
        sem_id_uniqueness_margin: float = 0.5,  # 语义ID唯一性约束边界值
    ) -> None:
        self._config = locals()
        
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.n_cat_feats = n_cat_features
        self.tag_alignment_weight = tag_alignment_weight
        self.tag_prediction_weight = tag_prediction_weight
        self.tag_embed_dim = tag_embed_dim  # 保存标签嵌入维度
        self.use_focal_loss = use_focal_loss  # 是否使用焦点损失
        self.focal_loss_params = focal_loss_params or {'gamma': 2.0}  # 焦点损失参数
        self.dropout_rate = dropout_rate  # 新增：Dropout率
        self.use_batch_norm = use_batch_norm  # 新增：是否使用BatchNorm
        self.alignment_temperature = alignment_temperature  # 新增：对比学习温度参数
        self.sem_id_uniqueness_weight = sem_id_uniqueness_weight  # 新增：语义ID唯一性约束权重
        
        # 如果未提供标签类别数量，则使用默认值
        if tag_class_counts is None:
            # 默认每层的标签类别数量，可以根据实际情况调整
            self.tag_class_counts = [10, 100, 1000][:n_layers]
        else:
            self.tag_class_counts = tag_class_counts[:n_layers]
        
        # 确保标签类别数量与层数匹配
        assert len(self.tag_class_counts) == n_layers, f"标签类别数量 {len(self.tag_class_counts)} 与层数 {n_layers} 不匹配"

        # 量化层
        self.layers = nn.ModuleList(modules=[
            Quantize(
                embed_dim=embed_dim,
                n_embed=codebook_size,
                forward_mode=codebook_mode,
                do_kmeans_init=codebook_kmeans_init,
                codebook_normalize=i == 0 and codebook_normalize,
                sim_vq=codebook_sim_vq,
                commitment_weight=commitment_weight
            ) for i in range(n_layers)
        ])

        # 标签预测器
        # 新增：每层拼接的embedding维度
        self.concat_embed_dims = [(embed_dim * (i + 1)) for i in range(n_layers)]

        # 标签预测器（输入维度变为拼接后的维度）
        # 兼容性处理: 尝试加载的权重的tag_class_counts可能与当前配置不同
        # 根据错误信息可以看出，存储的值为[7, 30, 97]，如果当前配置的值不同，则使用存储的值
        # 这样可以确保权重加载成功
        self._stored_tag_class_counts = None
        self.tag_predictors = nn.ModuleList(modules=[
            TagPredictor(
                embed_dim=self.concat_embed_dims[i],
                num_classes=self.tag_class_counts[i],
                hidden_dim=hidden_dims[0] // 2 * (i + 1),  # 根据层索引调整隐藏层维度
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                layer_idx=i
            ) for i in range(n_layers)
        ])
        
        # 标签嵌入投影层（输出维度变为拼接后的维度）
        self.tag_projectors = nn.ModuleList(modules=[
            nn.Sequential(
                nn.Linear(tag_embed_dim, hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dims[0], self.concat_embed_dims[i]),
                nn.LayerNorm(self.concat_embed_dims[i]) if codebook_normalize else nn.Identity()
            ) for i in range(n_layers)
        ])
        
        # 编码器和解码器
        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=embed_dim,
            normalize=codebook_normalize
        )

        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dims=hidden_dims[-1::-1],
            out_dim=input_dim,
            normalize=True
        )

        # 损失函数
        self.reconstruction_loss = (
            CategoricalReconstuctionLoss(n_cat_features) if n_cat_features != 0
            else ReconstructionLoss()
        )
        self.tag_alignment_loss = TagAlignmentLoss(
            alignment_weight=tag_alignment_weight,
            temperature=alignment_temperature
        )
        
        # 初始化标签预测损失，支持焦点损失
        self.tag_prediction_loss = TagPredictionLoss(
            use_focal_loss=use_focal_loss,
            focal_params=focal_loss_params,
            class_counts=None  # 将在训练过程中更新
        )
        
        # 添加语义ID唯一性约束损失
        self.sem_id_uniqueness_loss = SemanticIdUniquenessLoss(
            margin=sem_id_uniqueness_margin,
            weight=sem_id_uniqueness_weight
        )
        
        # 用于存储类别频率统计
        self.register_buffer('class_freq_counts', None)

    @cached_property
    def config(self) -> dict:
        return self._config
    
    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device
    
    def load_pretrained(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        
        # 检查权重文件中是否有与当前模型结构不匹配的情况
        model_dict = self.state_dict()
        pretrained_dict = state["model"]
        
        # 检查是否存在与标签预测器相关的不匹配
        tag_predictor_mismatch = False
        tag_class_counts_from_weights = []
        
        for i in range(self.n_layers):
            weight_key = f"tag_predictors.{i}.classifier.7.weight"
            if weight_key in pretrained_dict:
                pretrained_shape = pretrained_dict[weight_key].shape
                current_shape = model_dict[weight_key].shape if weight_key in model_dict else None
                
                if current_shape is not None and pretrained_shape[0] != current_shape[0]:
                    tag_predictor_mismatch = True
                    tag_class_counts_from_weights.append(pretrained_shape[0])
                else:
                    # 如果找不到对应层的权重或尺寸匹配，则保持当前的类别数
                    tag_class_counts_from_weights.append(self.tag_class_counts[i])
        
        # 检查tag_projectors相关的unexpected keys
        tag_projector_mismatch = False
        for i in range(self.n_layers):
            projector_key = f"tag_projectors.{i}.5.weight"
            if projector_key in pretrained_dict and projector_key not in model_dict:
                tag_projector_mismatch = True
                break
        
        # 如果有标签预测器不匹配，重新创建标签预测器以匹配权重文件
        if tag_predictor_mismatch and len(tag_class_counts_from_weights) == self.n_layers:
            print(f"检测到标签预测器不匹配，调整类别数量从 {self.tag_class_counts} 到 {tag_class_counts_from_weights}")
            self._stored_tag_class_counts = self.tag_class_counts  # 存储原始值
            self.tag_class_counts = tag_class_counts_from_weights  # 更新为权重文件中的值
            
            # 重新创建标签预测器
            self.tag_predictors = nn.ModuleList(modules=[
                TagPredictor(
                    embed_dim=self.concat_embed_dims[i],
                    num_classes=self.tag_class_counts[i],
                    hidden_dim=self._config.get('hidden_dims', [512, 256, 128])[0] // 2 * (i + 1),
                    dropout_rate=self._config.get('dropout_rate', 0.2),
                    use_batch_norm=self._config.get('use_batch_norm', True),
                    layer_idx=i
                ) for i in range(self.n_layers)
            ])
        
        # 如果有tag_projectors不匹配，重新创建tag_projectors以匹配权重文件
        if tag_projector_mismatch:
            print(f"检测到标签投影器不匹配，调整结构以匹配权重文件")
            # 重新创建tag_projectors，包含LayerNorm层
            self.tag_projectors = nn.ModuleList(modules=[
                nn.Sequential(
                    nn.Linear(self.tag_embed_dim, self._config.get('hidden_dims', [512, 256, 128])[0]),
                    nn.BatchNorm1d(self._config.get('hidden_dims', [512, 256, 128])[0]) if self._config.get('use_batch_norm', True) else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(self._config.get('dropout_rate', 0.2)),
                    nn.Linear(self._config.get('hidden_dims', [512, 256, 128])[0], self.concat_embed_dims[i]),
                    nn.LayerNorm(self.concat_embed_dims[i])  # 添加LayerNorm层，无论codebook_normalize如何设置
                ) for i in range(self.n_layers)
            ])
        
        # 处理剩余的不匹配键
        # 过滤掉那些在预训练模型中有但在当前模型中没有的键
        pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        # 如果过滤后的字典键数量少于原始字典，输出警告
        if len(pretrained_dict_filtered) < len(pretrained_dict):
            skipped_keys = set(pretrained_dict.keys()) - set(pretrained_dict_filtered.keys())
            print(f"警告: 以下键在加载时被跳过，因为当前模型结构中不存在: {skipped_keys}")
        
        try:
            # 尝试加载过滤后的权重
            model_dict.update(pretrained_dict_filtered)
            self.load_state_dict(model_dict)
            print(f"---Loaded HRQVAE Iter {state['iter']}---")
        except Exception as e:
            # 如果仍然失败，尝试宽松加载
            print(f"标准加载失败，尝试宽松加载: {str(e)}")
            self.load_state_dict(pretrained_dict, strict=False)
            print(f"---Loaded HRQVAE Iter {state['iter']} (宽松模式)---")
        
        # 如果有不匹配，输出警告
        if tag_predictor_mismatch:
            print(f"警告: 已自动调整标签预测器以匹配权重文件。原始类别数量: {self._stored_tag_class_counts}, 调整后: {self.tag_class_counts}")
        if tag_projector_mismatch:
            print(f"警告: 已自动调整标签投影器以匹配权重文件。")

    def encode(self, x: Tensor) -> Tensor:
        # 确保输入数据为 float32 以匹配模型权重
        x = x.float()
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def get_semantic_ids(
        self,
        encoded_x: Tensor,
        tags_emb: Optional[Tensor] = None,
        tags_indices: Optional[Tensor] = None,
        gumbel_t: float = 0.001
    ) -> HRqVaeOutput:
        """
        获取语义ID和计算相关损失
        
        参数:
            encoded_x: 编码后的输入特征
            tags_emb: 标签嵌入，形状为 [batch_size, n_layers, tag_embed_dim]
            tags_indices: 标签索引，形状为 [batch_size, n_layers]
            gumbel_t: Gumbel softmax 温度参数
            
        返回:
            HRqVaeOutput 包含嵌入、残差、语义ID和各种损失
        """
        res = encoded_x
        
        # 初始化损失
        quantize_loss = torch.tensor(0.0, device=encoded_x.device)  # 修改：初始化为张量
        tag_align_loss = torch.tensor(0.0, device=encoded_x.device)
        tag_pred_loss = torch.tensor(0.0, device=encoded_x.device)
        tag_pred_accuracy = torch.tensor(0.0, device=encoded_x.device)
        
        embs, residuals, sem_ids = [], [], []
        
        # 新增收集每层损失的列表
        tag_align_loss_by_layer = []
        tag_pred_loss_by_layer = []
        tag_pred_accuracy_by_layer = []
        
        for i, layer in enumerate(self.layers):
            residuals.append(res)
            quantized = layer(res, temperature=gumbel_t)
            quantize_loss = quantize_loss + quantized.loss  # 修改：确保损失累加
            emb, id = quantized.embeddings, quantized.ids
            
            # 添加当前层的embedding和id
            embs.append(emb)
            sem_ids.append(id)
            
            # 拼接前i+1层的embedding
            concat_emb = torch.cat(embs, dim=-1)  # [batch, (i+1)*embed_dim]
            
            # 如果提供了标签嵌入和索引，计算标签对齐损失和预测损失
            if tags_emb is not None and tags_indices is not None:
                # 获取当前层的标签嵌入和索引
                current_tag_emb = tags_emb[:, i]
                current_tag_idx = tags_indices[:, i]
                
                # 使用投影层将标签嵌入投影到与拼接后的码本嵌入相同的维度
                projected_tag_emb = self.tag_projectors[i](current_tag_emb)
                
                # 计算标签对齐损失
                align_loss = self.tag_alignment_loss(concat_emb, projected_tag_emb, i)
                tag_align_loss += align_loss.mean()
                tag_align_loss_by_layer.append(align_loss.mean())
                
                # 预测标签索引
                tag_logits = self.tag_predictors[i](concat_emb)
                pred_loss, pred_accuracy = self.tag_prediction_loss(tag_logits, current_tag_idx)
                
                tag_pred_loss += pred_loss
                tag_pred_accuracy += pred_accuracy
                tag_pred_loss_by_layer.append(pred_loss)
                tag_pred_accuracy_by_layer.append(pred_accuracy)
            
            # 更新残差
            res = res - emb
        
        # 如果没有提供标签数据，将标签相关损失设为0
        if tags_emb is None or tags_indices is None:
            tag_align_loss = torch.tensor(0.0, device=encoded_x.device)
            tag_pred_loss = torch.tensor(0.0, device=encoded_x.device)
            tag_pred_accuracy = torch.tensor(0.0, device=encoded_x.device)
        else:
            # 计算平均损失和准确率
            tag_align_loss = tag_align_loss / self.n_layers
            tag_pred_loss = tag_pred_loss / self.n_layers
            tag_pred_accuracy = tag_pred_accuracy / self.n_layers
            
            # 将按层的损失和准确率转换为张量
            tag_align_loss_by_layer = torch.stack(tag_align_loss_by_layer) if tag_align_loss_by_layer else None
            tag_pred_loss_by_layer = torch.stack(tag_pred_loss_by_layer) if tag_pred_loss_by_layer else None
            tag_pred_accuracy_by_layer = torch.stack(tag_pred_accuracy_by_layer) if tag_pred_accuracy_by_layer else None

        # 返回结果
        return HRqVaeOutput(
            embeddings=rearrange(embs, "b h d -> h d b"),
            residuals=rearrange(residuals, "b h d -> h d b"),
            sem_ids=rearrange(sem_ids, "b d -> d b"),
            quantize_loss=quantize_loss,
            tag_align_loss=tag_align_loss,
            tag_pred_loss=tag_pred_loss,
            tag_pred_accuracy=tag_pred_accuracy,
            # 添加以下三个新属性
            tag_align_loss_by_layer=tag_align_loss_by_layer,
            tag_pred_loss_by_layer=tag_pred_loss_by_layer,
            tag_pred_accuracy_by_layer=tag_pred_accuracy_by_layer
        )

    @torch.compile(mode="reduce-overhead")
    def forward(self, batch: SeqBatch, gumbel_t: float = 1.0) -> HRqVaeComputedLosses:
        x = batch.x
        
        # 获取标签嵌入和索引（如果有）
        tags_emb = getattr(batch, 'tags_emb', None)
        tags_indices = getattr(batch, 'tags_indices', None)
        
        # 确保输入数据为 float32 以匹配模型权重
        x = x.float()
        if tags_emb is not None:
            tags_emb = tags_emb.float()
        
        # 编码输入特征
        encoded_features = self.encode(x)

        # 获取语义ID和相关损失
        quantized = self.get_semantic_ids(encoded_features, tags_emb, tags_indices, gumbel_t)
        
        
        embs, residuals = quantized.embeddings, quantized.residuals
        # 解码,输入是所有层的嵌入之和，输出是重构的特征
        x_hat = self.decode(embs.sum(axis=-1)) 
        # print(f"x_hat形状: {x_hat.shape}")
        # 修复：处理分类特征的拼接
        x_hat = torch.cat([l2norm(x_hat[...,:-self.n_cat_feats]), x_hat[...,-self.n_cat_feats:]], axis=-1)
        # print(f"x_hat形状: {x_hat.shape}")
        # 计算重构损失
        reconstuction_loss = self.reconstruction_loss(x_hat, x)
        # print(f"重构损失: {reconstuction_loss.mean().item():.4f}")
        
        # 计算总损失
        rqvae_loss = quantized.quantize_loss
        tag_align_loss = quantized.tag_align_loss
        tag_pred_loss = quantized.tag_pred_loss
        tag_pred_accuracy = quantized.tag_pred_accuracy
        # print(f"RQVAE损失: {rqvae_loss.mean().item():.4f}")
        # print(f"标签对齐损失: {tag_align_loss.mean().item():.4f}")
        # print(f"标签预测损失: {tag_pred_loss.mean().item():.4f}")
        # print(f"标签预测准确率: {tag_pred_accuracy.mean().item():.4f}")
        
        # 新增：计算语义ID唯一性约束损失
        # 修复：正确处理张量形状转换
        # quantized.sem_ids 的形状是 [n_layers, batch_size]
        # 我们需要将其转换为 [batch_size, n_layers]
        sem_ids_tensor = quantized.sem_ids.transpose(0, 1)  # 直接转置，结果是 [batch_size, n_layers]
        sem_id_uniqueness_loss = self.sem_id_uniqueness_loss(sem_ids_tensor, encoded_features)
        
        # 总损失 = 重构损失 + RQVAE损失 + 标签对齐损失 + 标签预测损失 + 语义ID唯一性约束损失
        loss = (
            reconstuction_loss.mean() + 
            rqvae_loss.mean() + 
            self.tag_alignment_weight * tag_align_loss + 
            self.tag_prediction_weight * tag_pred_loss +
            self.sem_id_uniqueness_weight * sem_id_uniqueness_loss  # 新增：语义ID唯一性约束损失
        )
        # print(f"总损失: {loss.item():.4f}")

        with torch.no_grad():
            # 计算调试ID统计信息
            embs_norm = embs.norm(dim=1)
            p_unique_ids = (~torch.triu(
                (rearrange(quantized.sem_ids, "b d -> b 1 d") == rearrange(quantized.sem_ids, "b d -> 1 b d")).all(axis=-1), diagonal=1)
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]

       
        # 直接使用quantized中的按层损失
        tag_align_loss_by_layer = quantized.tag_align_loss_by_layer
        tag_pred_loss_by_layer = quantized.tag_pred_loss_by_layer
        tag_pred_accuracy_by_layer = quantized.tag_pred_accuracy_by_layer
        
        # 将语义ID唯一性约束损失添加到返回结果中
        return HRqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstuction_loss,
            rqvae_loss=rqvae_loss,
            tag_align_loss=tag_align_loss,
            tag_pred_loss=tag_pred_loss,
            tag_pred_accuracy=tag_pred_accuracy,
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
            # 新增按层的损失和准确率
            tag_align_loss_by_layer=tag_align_loss_by_layer,
            tag_pred_loss_by_layer=tag_pred_loss_by_layer,
            tag_pred_accuracy_by_layer=tag_pred_accuracy_by_layer,
            # 新增语义ID唯一性约束损失
            sem_id_uniqueness_loss=sem_id_uniqueness_loss
        )
    
    def predict_tags(self, x: Tensor, gumbel_t: float = 0.001) -> Dict[str, Tensor]:
        """
        预测输入特征对应的标签索引
        
        参数:
            x: 输入特征，形状可以是 [batch_size, feature_dim] 或 [batch_size, seq_len, feature_dim]
            gumbel_t: Gumbel softmax 温度参数
            
        返回:
            包含预测标签索引和置信度的字典
        """
        # 检查输入维度并处理序列数据
        original_shape = x.shape
        if len(original_shape) == 3:
            # 输入是序列数据 [batch_size, seq_len, feature_dim]
            batch_size, seq_len, feature_dim = original_shape
            # 将序列展平为 [batch_size*seq_len, feature_dim]
            x = x.reshape(-1, feature_dim)
            is_sequence = True
        else:
            # 输入已经是 [batch_size, feature_dim]
            is_sequence = False
            
        # 编码输入特征
        res = self.encode(x)
        print(f"编码后的特征形状: {res.shape}")
        
        tag_predictions = []
        tag_confidences = []
        embs = []  # 存储每层的embedding用于拼接
        
        # 对每一层进行预测
        for i, layer in enumerate(self.layers):
            # 获取量化嵌入
            quantized = layer(res, temperature=gumbel_t)
            emb = quantized.embeddings
            
            # 添加当前层embedding
            embs.append(emb)
            
            # 拼接前i+1层的embedding
            concat_emb = torch.cat(embs, dim=-1)
            
            # 使用拼接后的embedding预测标签
            tag_logits = self.tag_predictors[i](concat_emb)
            tag_probs = torch.softmax(tag_logits, dim=-1)
            
            # 获取最可能的标签索引及其置信度
            confidence, prediction = torch.max(tag_probs, dim=-1)
            
            tag_predictions.append(prediction)
            tag_confidences.append(confidence)
            
            # 更新残差
            res = res - emb
        
        # 将预测结果重新整形为原始序列形状（如果输入是序列）
        if is_sequence:
            tag_predictions = [pred.reshape(batch_size, seq_len) for pred in tag_predictions]
            tag_confidences = [conf.reshape(batch_size, seq_len) for conf in tag_confidences]
        
        return {
            "predictions": torch.stack(tag_predictions, dim=-1),  # 对于序列: [batch_size, seq_len, n_layers]
            "confidences": torch.stack(tag_confidences, dim=-1)   # 对于序列: [batch_size, seq_len, n_layers]
        }

    def update_class_counts(self, class_counts_dict):
        """
        更新类别频率计数
        
        参数:
            class_counts_dict: 包含每层类别计数的字典
        """
        # 将字典转换为模块字典，而不是直接赋值
        for layer_idx, counts in class_counts_dict.items():
            # 确保counts是张量
            if not isinstance(counts, torch.Tensor):
                counts = torch.tensor(counts, device=self.device)
            # 使用register_buffer动态注册，或更新已有的buffer
            self.register_buffer(f'class_freq_counts_{layer_idx}', counts)
        
        # 存储层索引列表，以便后续使用
        self.class_freq_layers = list(class_counts_dict.keys())
    
   