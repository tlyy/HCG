import flax.linen as nn
import jax.numpy as jnp



class DyT(nn.Module):
    num_features: int
    alpha_init_value: float = 0.5

    def setup(self):
        self.alpha = self.param('alpha', nn.initializers.constant(self.alpha_init_value), (1,))
        self.weight = self.param('weight', nn.initializers.constant(1), (self.num_features,))
        self.bias = self.param('bias', nn.initializers.constant(0), (self.num_features,))

    def __call__(self, x):
        x = nn.tanh(self.alpha * x)
        return x * self.weight + self.bias



class TransformerBlock(nn.Module):
    emb_dim: int
    num_heads: int
    mlp_dim: int
    seq_len: int

    def setup(self):
        self.ln1 = nn.LayerNorm()
        self.mha = nn.SelfAttention(num_heads=self.num_heads)
        self.ln2 = nn.LayerNorm()
        self.mlp = nn.Sequential([
            nn.Dense(self.mlp_dim),
            nn.gelu,
            nn.Dense(self.emb_dim)
        ])
        self.mask = jnp.tril(jnp.ones((self.seq_len, self.seq_len)))
        # self.dyt1 = DyT(self.emb_dim)
        # self.dyt2 = DyT(self.emb_dim)

    def __call__(self, x):        
        # Pre-LN 结构
        y = self.ln1(x)
        # y = self.dyt1(x)
        y = self.mha(y, mask=self.mask)
        y += x
        
        z = self.ln2(y)
        # z = self.dyt2(y)
        z = self.mlp(z)
        z += y
        return z, z


class TransformerTM(nn.Module):
    emb_dim: int
    num_heads: int
    num_layers: int
    num_actions: int
    seq_len: int
    dropout: float = 0.1
    mlp_expansion: int = 4  # 新增 MLP 扩展系数

    def setup(self):
        self.pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (self.seq_len, self.emb_dim)
        )
        self.transformer_blocks = nn.scan(
            TransformerBlock,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            length=self.num_layers
        )(
            emb_dim=self.emb_dim,
            num_heads=self.num_heads,
            mlp_dim=self.emb_dim,
            seq_len=self.seq_len,
        )
        self.ln = nn.LayerNorm()
        self.action_emb = nn.Dense(self.emb_dim)
        self.state_emb = nn.Dense(self.emb_dim)
        self.state_decoder = nn.Dense(11**2*128)

    def __call__(self, x, actions):
        # x.shape = [11, 11, 128]
        # 状态嵌入
        state_emb = self.state_emb(x.reshape(1, -1))  # [1, emb_dim]
        # 动作嵌入
        action_onehot = nn.one_hot(actions, self.num_actions)  # [5, num_actions]
        action_embs = self.action_emb(action_onehot)  # [5, emb_dim]
        
        # 拼接序列
        x = jnp.concatenate([state_emb, action_embs], axis=0)  # [6, emb_dim]
        x += self.pos_embedding  # 广播添加位置编码

        # Transformer 处理
        x, _ = self.transformer_blocks(x)
        # 解码输出
        x = x[1:]  # 取后5个位置的输出 [5, emb_dim]
        x = self.state_decoder(x)  # [5, 11*11*128]
        return x.reshape(-1, 11, 11, 128)
