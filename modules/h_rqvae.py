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
        dropout_rate = min(0.4, dropout_rate + layer_idx * 0.05)
        
        # 自注意力机制 - 帮助模型关注输入特征中的重要部分
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim),
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
        mid_dim = hidden_dim // (layer_idx + 1) if layer_idx > 0 else hidden_dim
        
        # 第二部分：残差模块
        self.residual_block1 = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mid_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_batch_norm else nn.Identity(),
        )
        
        # 第三部分：残差模块2
        self.residual_block2 = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mid_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_batch_norm else nn.Identity(),
        )
        
        # 输出层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # 输出层降低dropout以保留更多信息
            nn.Linear(mid_dim, num_classes)
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
        self.load_state_dict(state["model"])
        print(f"---Loaded HRQVAE Iter {state['iter']}---")

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def get_semantic_ids(
        self,
        x: Tensor,
        tags_emb: Optional[Tensor] = None,
        tags_indices: Optional[Tensor] = None,
        gumbel_t: float = 0.001
    ) -> HRqVaeOutput:
        """
        获取语义ID和计算相关损失
        
        参数:
            x: 输入特征
            tags_emb: 标签嵌入，形状为 [batch_size, n_layers, tag_embed_dim]
            tags_indices: 标签索引，形状为 [batch_size, n_layers]
            gumbel_t: Gumbel softmax 温度参数
            
        返回:
            HRqVaeOutput 包含嵌入、残差、语义ID和各种损失
        """
        res = self.encode(x)
        
        # 初始化损失
        quantize_loss = torch.tensor(0.0, device=x.device)  # 修改：初始化为张量
        tag_align_loss = torch.tensor(0.0, device=x.device)
        tag_pred_loss = torch.tensor(0.0, device=x.device)
        tag_pred_accuracy = torch.tensor(0.0, device=x.device)
        
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
            tag_align_loss = torch.tensor(0.0, device=x.device)
            tag_pred_loss = torch.tensor(0.0, device=x.device)
            tag_pred_accuracy = torch.tensor(0.0, device=x.device)
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
        
        # 获取语义ID和相关损失
        quantized = self.get_semantic_ids(x, tags_emb, tags_indices, gumbel_t)
        
        
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
        
        # 总损失 = 重构损失 + RQVAE损失 + 标签对齐损失 + 标签预测损失
        loss = (
            reconstuction_loss.mean() + 
            rqvae_loss.mean() + 
            self.tag_alignment_weight * tag_align_loss + 
            self.tag_prediction_weight * tag_pred_loss
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
            tag_pred_accuracy_by_layer=tag_pred_accuracy_by_layer
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
    
   