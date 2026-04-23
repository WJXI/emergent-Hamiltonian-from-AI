import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 骨干网络：Pairformer Block
# ==========================================
class PairformerBlock(nn.Module):
    def __init__(self, c_s=64, c_z=32, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        
        self.s_to_z = nn.Linear(c_s * 2, c_z)
        self.z_mlp = nn.Sequential(nn.Linear(c_z, c_z * 2), nn.GELU(), nn.Linear(c_z * 2, c_z))
        
        self.qkv = nn.Linear(c_s, 3 * c_s)
        self.z_to_bias = nn.Linear(c_z, num_heads)
        self.attn_out = nn.Linear(c_s, c_s)
        
        self.norm_s = nn.LayerNorm(c_s)
        self.norm_z = nn.LayerNorm(c_z)

    def forward(self, s, z, mask=None):
        B, N, _ = s.shape
        # 1. Update Pair Rep (z)
        s_i = s.unsqueeze(2).expand(B, N, N, -1)
        s_j = s.unsqueeze(1).expand(B, N, N, -1)
        z = z + self.s_to_z(torch.cat([s_i, s_j], dim=-1))
        z = z + self.z_mlp(self.norm_z(z))
        
        # 2. Update Single Rep (s) via Pair-biased Attention
        q, k, v = map(lambda t: t.view(B, N, self.num_heads, -1).transpose(1, 2), self.qkv(self.norm_s(s)).chunk(3, dim=-1))
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        attn_logits = attn_logits + self.z_to_bias(z).permute(0, 3, 1, 2) # Inject Physics Bias
        
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.view(B, 1, 1, N), -1e9)
            
        out = torch.matmul(F.softmax(attn_logits, dim=-1), v).transpose(1, 2).contiguous().view(B, N, -1)
        s = s + self.attn_out(out)
        return s, z

# ==========================================
# 2. 主网络：【公理一】等变聚合与切空间投影
# ==========================================
class MiniAF3ScoreModel(nn.Module):
    def __init__(self, c_s=64, c_z=32, num_blocks=4):
        super().__init__()
        self.seq_emb = nn.Embedding(2, c_s)
        self.pos_emb = nn.Embedding(100, c_s)
        self.time_emb = nn.Sequential(nn.Linear(1, c_s), nn.GELU(), nn.Linear(c_s, c_s))
        
        self.dist_emb = nn.Embedding(100, c_z)
        self.inner_prod_proj = nn.Linear(1, c_z)
        
        self.blocks = nn.ModuleList([PairformerBlock(c_s, c_z) for _ in range(num_blocks)])
        
        # 【公理一核心】预测标量权重 W_ij，而非直接预测 3D 向量！
        self.weight_head = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, c_z),
            nn.GELU(),
            nn.Linear(c_z, 1)
        )

    def forward(self, seq, spins, t, mask):
        B, N = seq.shape
        device = seq.device
        
        positions = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        s = self.seq_emb(seq) + self.pos_emb(positions) + self.time_emb(t).unsqueeze(1)
        
        dist = torch.abs(positions.unsqueeze(2) - positions.unsqueeze(1)).clamp(max=99)
        z = self.dist_emb(dist) + self.inner_prod_proj(torch.einsum('bid,bjd->bij', spins, spins).unsqueeze(-1))
        
        for block in self.blocks:
            s, z = block(s, z, mask)
            
        # 1. 提取交互权重 W_ij
        W = self.weight_head(z).squeeze(-1) # (B, N, N)
        
        # 【物理约束】作用力相互作用必须满足牛顿第三定律 (对称性 J_ij = J_ji)
        W = (W + W.transpose(1, 2)) / 2.0
        
        # ------------------- 修改开始 -------------------
        # 【物理约束】自旋不会对自己施加力 (对角线清零)
        # 生成一个 N x N 的单位矩阵，取反后变成对角线为 0，其他为 1 的掩码
        diag_mask = 1.0 - torch.eye(N, device=device).unsqueeze(0) # (1, N, N)
        W = W * diag_mask
        # ------------------- 修改结束 -------------------
        
        # 屏蔽 Padding 的虚假力
        W = W * mask.float().unsqueeze(1) * mask.float().unsqueeze(2)
        
        # 2. 等变聚合 (Equivariant Aggregation) -> \sum W_ij * S_j
        raw_score = torch.bmm(W, spins) # (B, N, 3)
        
        # 3. 严格切空间投影 (Tangent Space Projection)
        radial_comp = torch.sum(raw_score * spins, dim=-1, keepdim=True)
        tangent_score = raw_score - radial_comp * spins
        
        return tangent_score * mask.unsqueeze(-1)

# ==========================================
# 3. 黎曼 SDE：【公理二】逆指数映射 (Log Map)
# ==========================================
def sample_spherical_noise_and_target(S_0, t_batch):
    """
    基于黎曼几何生成加噪点 S_t，并返回存在于 S_t 切平面内的严格目标 Score
    """
    # 1. 切向随机游走 (生成 S_t)
    v = torch.randn_like(S_0)
    v_tan = v - torch.sum(v * S_0, dim=-1, keepdim=True) * S_0
    v_norm = torch.norm(v_tan, dim=-1, keepdim=True) + 1e-8
    u = v_tan / v_norm
    theta = v_norm * torch.sqrt(t_batch) # 布朗运动弧长
    S_t = S_0 * torch.cos(theta) + u * torch.sin(theta)
    
    # 2. 【公理二核心】计算 Target Score = (1/t) * Log_{S_t}(S_0)
    # 确保 Target 向量绝对位于 S_t 的切平面上！
    cos_angle = torch.sum(S_0 * S_t, dim=-1, keepdim=True).clamp(-0.9999, 0.9999)
    theta_true = torch.acos(cos_angle)
    
    # 解决 theta 趋近 0 时的数值奇异性
    coef = torch.where(theta_true < 1e-4, 1.0, theta_true / torch.sin(theta_true))
    
    # 黎曼对数映射公式: Log_x(y) = (theta / sin(theta)) * (y - cos(theta)*x)
    log_map = coef * (S_0 - cos_angle * S_t)
    
    target_score = log_map / t_batch
    
    return S_t, target_score