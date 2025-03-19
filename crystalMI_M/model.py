import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# GPT Configuration Class
class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd,
                 embd_pdrop, resid_pdrop, attn_pdrop):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop

# Trainer Configuration Class
class TrainerConfig:
    def __init__(self, max_epochs, batch_size, learning_rate, betas, weight_decay, eval_interval, ckpt_path=None):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.eval_interval = eval_interval
        self.ckpt_path = ckpt_path  # Path for saving/loading model

# Causal Self-Attention Module
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads."
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Causal mask to ensure attention is only to the left
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                      .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)  # (B, T, 3 * C)
        qkv = qkv.view(B, T, self.n_head, 3 * self.head_dim).transpose(1, 2)  # (B, n_head, T, 3 * head_dim)
        q, k, v = qkv.split(self.head_dim, dim=-1)  # Each is (B, n_head, T, head_dim)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_head, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y

# Transformer Block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# GPT Model
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding layers
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.embd_pdrop)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output layer
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Input sequence length exceeds block size."

        # Get embeddings
        token_embeddings = self.token_embedding(idx)  # (B, T, n_embd)
        position_embeddings = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = self.dropout(token_embeddings + position_embeddings)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Output logits
        logits = self.head(x)  # (B, T, vocab_size)

        # Compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# Utility functions
def load_model(model_path, config, device):
    model = GPT(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def generate_target(model, encoding_text, ctoi, itoc, device, max_new_tokens=50, temperature=1.0, do_sample=False, top_k=None):
    model.eval()
    idx = torch.tensor([[ctoi.get(c, ctoi['<unk>']) for c in encoding_text + '\n']], dtype=torch.long).to(device)
    generated_idx = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample, top_k=top_k)
    generated_text = ''.join([itoc[i] for i in generated_idx[0].tolist()])
    target_start = generated_text.find('\n') + 1
    target_end = generated_text.find('\n', target_start)
    if target_end == -1:
        target_text = generated_text[target_start:]
    else:
        target_text = generated_text[target_start:target_end]
    return target_text.strip()
