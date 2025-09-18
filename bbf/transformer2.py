import jax
import flax.linen as nn
import jax.numpy as jnp
from typing import Optional, Tuple



class NativeSparseAttention(nn.Module):
    num_heads: int
    window_size: int       # 局部窗口大小
    embed_dim: int
    dropout_rate: float = 0.1

    def setup(self):
        self.query_dense = nn.Dense(self.embed_dim)
        self.key_dense = nn.Dense(self.embed_dim)
        self.value_dense = nn.Dense(self.embed_dim)
        self.output_dense = nn.Dense(self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        batch_size, seq_len, _ = x.shape

        # 1. 投影Q/K/V
        queries = self.query_dense(x)  # [B, seq_len, embed_dim]
        keys = self.key_dense(x)
        values = self.value_dense(x)

        # 2. 分割多头
        head_dim = self.embed_dim // self.num_heads
        queries = queries.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # 3. 生成稀疏注意力掩码（局部窗口）
        mask = jnp.tril(jnp.ones((seq_len, seq_len)), self.window_size)  # 下三角+窗口
        mask = mask[None, None, ...]  # [1, 1, seq_len, seq_len]

        # 4. 计算注意力分数
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', queries, keys)
        attn_scores = attn_scores / jnp.sqrt(head_dim)
        attn_scores = jnp.where(mask, attn_scores, -jnp.inf)  # 掩码无关位置
        attn_weights = nn.softmax(attn_scores, axis=-1)

        # 5. 加权聚合值
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, values)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        # 6. 输出投影
        output = self.output_dense(attn_output)
        return self.dropout(output, deterministic=not training)



class MultiHeadLatentAttention(nn.Module):
    num_heads: int          # 注意力头数
    latent_dim: int         # 潜在变量的维度
    embed_dim: int          # 输入嵌入维度

    def setup(self):
        # 输入序列的键/值投影层
        self.key_dense = nn.Dense(self.embed_dim)
        self.value_dense = nn.Dense(self.embed_dim)
        # 输出投影层
        self.output_dense = nn.Dense(self.embed_dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size = x.shape[0]

        # 1. 投影键和值
        keys = self.key_dense(x)    # [seq_len, embed_dim]
        values = self.value_dense(x)

        # 2. 分割多头（键/值）
        keys = keys.reshape(-1, self.num_heads, self.embed_dim // self.num_heads)
        values = values.reshape(-1, self.num_heads, self.embed_dim // self.num_heads)
        keys = keys.transpose(1, 0, 2)   # [num_heads, seq_len, head_dim]
        values = values.transpose(1, 0, 2)

        # 3. 潜在查询（广播到批次维度）
        latent_queries = jnp.tile(
            self.latent_queries[None, ...],  # [num_heads, latent_dim, head_dim]
            (1, 1, 1)
        )  # [num_heads, latent_dim, head_dim]

        # 4. 计算注意力分数
        attn_scores = jnp.einsum('hqd,hkd->hqk', latent_queries, keys)
        attn_weights = nn.softmax(attn_scores / jnp.sqrt(keys.shape[-1]), axis=-1)

        # 5. 加权聚合值向量
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, values)
        attn_output = attn_output.transpose(0, 2, 1, 3)  # [B, latent_dim, num_heads, head_dim]
        attn_output = attn_output.reshape(batch_size, self.latent_dim, self.embed_dim)

        # 6. 输出投影
        output = self.output_dense(attn_output)
        return output



class RMSNorm(nn.Module):
    dim: int
    eps: float

    def setup(self):
        self.weight = self.param('weight', nn.initializers.ones, (self.dim,))

    def __call__(self, x):
        mean_squared = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        rms = jnp.sqrt(mean_squared + self.eps)
        return rms



def apply_rotary_emb(x: jnp.ndarray, freqs_cis: jnp.ndarray) -> jnp.ndarray:
    """
    Applies rotary positional embeddings to the input tensor (JAX version).

    Args:
        x (jnp.ndarray): Input tensor of shape [B, S, H] or [B, S, N, H]
        freqs_cis (jnp.ndarray): Precomputed complex exponentials of shape [S, H//2]

    Returns:
        jnp.ndarray: Tensor with rotary embeddings applied
    """
    # Convert input to complex numbers
    shape = x.shape
    x_complex = jnp.dstack((x, jnp.zeros_like(x))).reshape(*shape[:-1], -1, 2)
    x_complex = jax.lax.complex(x_complex[..., 0], x_complex[..., 1])

    # Reshape frequency tensor for broadcasting
    freqs_cis = freqs_cis.reshape(1, x.shape[1], 1, -1)

    # Apply rotation using complex multiplication
    rotated = x_complex * freqs_cis

    # Convert back to real numbers
    real = jnp.real(rotated)
    imag = jnp.imag(rotated)
    output = jnp.stack([real, imag], axis=-1).reshape(*shape)

    return output.astype(x.dtype)



class MLA(nn.Module):
    dim: int
    n_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    qk_head_dim: int
    v_head_dim: int
    max_seq_len: int = 6
    # attn_impl: str = "naive"
    
    def setup(self):
        self.n_local_heads = self.n_heads  # Assuming single device, modify if distributed
        
        # Query projections
        self.wq_a = nn.Dense(self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = nn.Dense(self.n_heads * self.qk_head_dim)
        
        # Key/Value projections
        self.wkv_a = nn.Dense(self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = nn.Dense(self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        
        # Output projection
        self.wo = nn.Dense(self.dim)
        
        # Softmax scale
        self.softmax_scale = self.qk_head_dim ** -0.5


    def __call__(self, 
                x: jnp.ndarray, 
                start_pos: int, 
                freqs_cis: jnp.ndarray, 
                mask: Optional[jnp.ndarray] = None,
                cache: Optional[dict] = None) -> Tuple[jnp.ndarray, dict]:
        print(x.shape)
        bsz, seqlen, _ = x.shape
        end_pos = start_pos + seqlen
        
        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_a(x)
            q = self.q_norm(q)
            q = self.wq_b(q)
        
        q = q.reshape(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = jnp.split(q, [self.qk_nope_head_dim], axis=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        
        # Key/Value projection
        kv = self.wkv_a(x)
        kv, k_pe = jnp.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], axis=-1)
        k_pe = apply_rotary_emb(k_pe[:, :, None, :], freqs_cis)  # Add head dimension
        
        # # Initialize cache if not provided
        # if cache is None:
        #     if self.attn_impl == "naive":
        #         cache = {
        #             'k': jnp.zeros((self.max_batch_size, max_seq_len, 
        #                           self.n_local_heads, self.self.qk_head_dim)),
        #             'v': jnp.zeros((self.max_batch_size, max_seq_len,
        #                           self.n_local_heads, self.self.v_head_dim))
        #         }
        #     else:
        #         cache = {
        #             'kv': jnp.zeros((self.max_batch_size, max_seq_len, self.kv_lora_rank)),
        #             'pe': jnp.zeros((self.max_batch_size, max_seq_len, self.qk_rope_head_dim))
        #         }

        if self.attn_impl == "naive":
            # Process naive attention implementation
            q = jnp.concatenate([q_nope, q_pe], axis=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.reshape(bsz, seqlen, self.n_local_heads, -1)
            k_nope, v = jnp.split(kv, [self.qk_nope_head_dim, self.v_head_dim], axis=-1)
            k = jnp.concatenate([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], axis=-1)
            self.k_cache[:bsz, start_pos: end_pos] = k
            self.v_cache[:bsz, start_pos: end_pos] = v
            scores = jnp.einsum('bshd,bthd->bsht', q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        
            
            # # Update cache
            # k_cache = jax.lax.dynamic_update_slice(
            #     cache['k'], 
            #     k, 
            #     (0, start_pos, 0, 0)
            # )
            # v_cache = jax.lax.dynamic_update_slice(
            #     cache['v'],
            #     v,
            #     (0, start_pos, 0, 0)
            # )
            
            # # Compute attention
            # k_all = jax.lax.dynamic_slice(
            #     k_cache,
            #     (0, 0, 0, 0),
            #     (bsz, end_pos, self.n_local_heads, self.self.qk_head_dim)
            # )
            # scores = jnp.einsum('bshd,bthd->bsht', q, k_all) * self.softmax_scale
            
        else:
            # Process optimized attention implementation
            wkv_b = self.wkv_b.variables['params']['kernel'] if self.wkv_b.scale is None else weight_dequant(self.wkv_b.variables['params']['kernel'])
            wkv_b = wkv_b.reshape(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = jnp.einsum('bshd,hdc->bshc', q_nope, wkv_b[:, :self.qk_nope_head_dim])
            
            self.kv_cache[:bsz, start_pos, end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos, end_pos] = k_pe

            scores = (
                jnp.einsum('bshc,btc->bsht', q_nope, self.kv_cache[:bsz, :end_pos]) +
                jnp.einsum('bshr,btr->bsht', q_pe, self.pe_cache[:bsz, :end_pos])
            ) * self.softmax_scale

            # # Update cache
            # kv_normalized = self.kv_norm(kv)
            # kv_cache = jax.lax.dynamic_update_slice(
            #     cache['kv'],
            #     kv_normalized,
            #     (0, start_pos, 0)
            # )
            # pe_cache = jax.lax.dynamic_update_slice(
            #     cache['pe'],
            #     k_pe.squeeze(2),
            #     (0, start_pos, 0)
            # )
            
            # # Compute attention scores
            # kv_all = jax.lax.dynamic_slice(
            #     kv_cache,
            #     (0, 0, 0),
            #     (bsz, end_pos, self.self.kv_lora_rank)
            # )
            # pe_all = jax.lax.dynamic_slice(
            #     pe_cache,
            #     (0, 0, 0),
            #     (bsz, end_pos, self.self.qk_rope_head_dim)
            # )
            # scores = (
            #     jnp.einsum('bshc,btc->bsht', q_nope, kv_all) +
            #     jnp.einsum('bshr,btr->bsht', q_pe, pe_all)
            # ) * self.softmax_scale

        # Apply mask if provided
        if mask is not None:
            scores += mask[:, None, :, :]

        # Softmax
        scores = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)

        # Compute output
        if self.attn_impl == "naive":
            v_all = jax.lax.dynamic_slice(
                v_cache,
                (0, 0, 0, 0),
                (bsz, end_pos, self.n_local_heads, self.self.v_head_dim)
            )
            x = jnp.einsum('bsht,bthd->bshd', scores, v_all)
            new_cache = {'k': k_cache, 'v': v_cache}
        else:
            x = jnp.einsum('bsht,btc->bshc', scores, kv_all)
            x = jnp.einsum('bshc,hdc->bshd', x, wkv_b[:, -self.self.v_head_dim:])
            new_cache = {'kv': kv_cache, 'pe': pe_cache}

        # Final projection
        x = self.wo(x.reshape(bsz, seqlen, -1))
        return x, new_cache



class DyT(nn.Module):
    num_features: int
    alpha_init_value: float = 0.5

    def setup(self):
        self.alpha = self.param('alpha', nn.initializers.ones()*self.alpha_init_value, (1,))
        self.weight = self.param('alpha', nn.initializers.ones(), (self.num_features,))
        self.bias = self.param('alpha', nn.initializers.zeros(), (self.num_features,))

    def __call__(self, x):
        x = nn.tanh(self.alpha * x)
        return x * self.weight + self.bias



class TransformerBlock(nn.Module):
    emb_dim: int
    num_heads: int
    mlp_dim: int
    dropout: float = 0.1

    def setup(self):
        self.ln1 = nn.LayerNorm()
        self.mha = nn.SelfAttention(num_heads=self.num_heads)
        self.ln2 = nn.LayerNorm()
        self.mlp = nn.Sequential([
            nn.Dense(self.mlp_dim),
            nn.gelu,
            nn.Dense(self.emb_dim)
        ])
        self.mask = jnp.tril(jnp.ones((6, 6)))


    def __call__(self, x):
        # 第一层归一化
        y = self.ln1(x)
        # 多头自注意力机制
        y = self.mha(y, self.mask)
        # 残差连接
        y += x
        # 第二层归一化
        z = self.ln2(y)
        # 多层感知机
        z = self.mlp(z)
        # 最终残差连接
        return z + y



class VisionTransformer(nn.Module):
    emb_dim: int
    num_heads: int
    num_layers: int
    num_actions: int
    dropout: float = 0.1

    def setup(self):
        # 位置编码
        self.pos_embedding = self.param('pos_embedding', nn.initializers.normal(stddev=0.02), (6, self.emb_dim))
        # 使用 nn.scan 定义多层 TransformerBlock
        self.transformer_blocks = [TransformerBlock(self.emb_dim, self.num_heads, self.emb_dim, self.dropout) for _ in range(self.num_layers)]
        # 最后一层归一化
        self.ln = nn.LayerNorm()
        # 动作嵌入层
        self.action_emb = nn.Dense(self.emb_dim)
        self.state_emb = nn.Dense(self.emb_dim)
        self.state_ori = nn.Dense(11**2*128)


    def __call__(self, x, actions):
        # 将动作转换为 one-hot 编码
        action_onehot = nn.one_hot(actions, self.num_actions)
        # 动作嵌入
        action_embs = self.action_emb(action_onehot).reshape(-1, self.emb_dim)
        # 拼接输入和动作嵌入，并添加位置编码
        x = self.state_emb(x.reshape(1, -1))
        x = jnp.concatenate([x, action_embs]) + self.pos_embedding
        # 通过多层 TransformerBlock
        for block in self.transformer_blocks:
            x = block(x)
        # 取最后一个输出并截取前 128 个元素
        x = x[1:]
        x = self.state_ori(x)
        # 调整输出形状
        return x.reshape(5, 11, 11, 128)