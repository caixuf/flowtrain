#!/usr/bin/env python3
"""
FlowTrain: TinyMLA 故事小模型快速预训练脚本
使用本地 NVIDIA RTX 5060 GPU 训练，并导出为 FlowServe 兼容的 FLSV 二进制模型格式。
"""

import math
import struct
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. 结构与超参数配置 (与 flowserve TransformerConfig 完全对齐)
# ============================================================================
VOCAB_SIZE = 259       # 3 特殊标记 (BOS=0, EOS=1, PAD=2) + 256 单字节
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 4
HEAD_DIM = 16
MLA_LATENT_DIM = 16    # 4x 压缩 (4*16 -> 16)
INTERMEDIATE_DIM = 128
MAX_SEQ_LEN = 64
BATCH_SIZE = 32
STEPS = 600
LR = 3e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 2. Byte-level 分词器
# ============================================================================
class ByteTokenizer:
    BOS = 0
    EOS = 1
    PAD = 2
    BYTE_OFFSET = 3

    def encode(self, text: str) -> list[int]:
        tokens = [self.BOS]
        tokens.extend([b + self.BYTE_OFFSET for b in text.encode("utf-8")])
        tokens.append(self.EOS)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        bytes_list = []
        for t in tokens:
            if t >= self.BYTE_OFFSET and t < self.BYTE_OFFSET + 256:
                bytes_list.append(t - self.BYTE_OFFSET)
        return bytes(bytes_list).decode("utf-8", errors="replace")

tokenizer = ByteTokenizer()

# ============================================================================
# 3. 故事小数据集 (精选简单童话短语料)
# ============================================================================
STORIES = [
    "Once upon a time, there was a little cat named Mia. She liked to run and play in the sun.",
    "Once upon a time, a little puppy found a red ball in the garden. He was very happy and barked.",
    "Once upon a time, there lived a friendly bear who loved sweet honey and big warm hugs.",
    "Once upon a time, a tiny bird learned how to fly over the green trees and blue sky.",
    "Once upon a time, Lily had a magic flower that would glow brightly in the dark night.",
    "Once upon a time, Tom and his dog went to the big river. They saw pretty fish jumping.",
    "Once upon a time, a small mouse shared his cheese with an owl. They became good friends.",
    "Once upon a time, the sun came up and smiled at the flowers. All the birds began to sing.",
    "Once upon a time, a brave bunny jumped over the river and found a garden of sweet carrots.",
    "Once upon a time, there was a shiny star that twinkled happily for all the sleeping children.",
    "Once upon a time, Tim found a big green apple on the grass. It was sweet and crunchy.",
    "Once upon a time, a little blue car drove fast on the road to visit the beautiful beach.",
]

def get_batch():
    batch_tokens = []
    for _ in range(BATCH_SIZE):
        story = STORIES[torch.randint(0, len(STORIES), (1,)).item()]
        tokens = tokenizer.encode(story)
        if len(tokens) > MAX_SEQ_LEN:
            tokens = tokens[:MAX_SEQ_LEN]
        else:
            tokens = tokens + [ByteTokenizer.PAD] * (MAX_SEQ_LEN - len(tokens))
        batch_tokens.append(tokens)
    x = torch.tensor(batch_tokens, dtype=torch.long, device=DEVICE)
    return x[:, :-1], x[:, 1:]

# ============================================================================
# 4. PyTorch MLA Transformer 模型定义
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class MLALayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_dkv = nn.Linear(HIDDEN_DIM, MLA_LATENT_DIM, bias=False)
        self.w_dq = nn.Linear(HIDDEN_DIM, MLA_LATENT_DIM, bias=False)
        self.w_uk = nn.Linear(MLA_LATENT_DIM, NUM_HEADS * HEAD_DIM, bias=False)
        self.w_uv = nn.Linear(MLA_LATENT_DIM, NUM_HEADS * HEAD_DIM, bias=False)
        self.w_uq = nn.Linear(MLA_LATENT_DIM, NUM_HEADS * HEAD_DIM, bias=False)
        self.w_o = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_DIM, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        c_kv = self.w_dkv(x)       # [B, T, L]
        c_q = self.w_dq(x)         # [B, T, L]

        K = self.w_uk(c_kv).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        V = self.w_uv(c_kv).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        Q = self.w_uq(c_q).view(B, T, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        scale = 1.0 / math.sqrt(HEAD_DIM)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, NUM_HEADS * HEAD_DIM)
        return self.w_o(out)

class SwiGLUFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_gate = nn.Linear(HIDDEN_DIM, INTERMEDIATE_DIM, bias=False)
        self.w_up = nn.Linear(HIDDEN_DIM, INTERMEDIATE_DIM, bias=False)
        self.w_down = nn.Linear(INTERMEDIATE_DIM, HIDDEN_DIM, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(HIDDEN_DIM)
        self.mla = MLALayer()
        self.norm2 = RMSNorm(HIDDEN_DIM)
        self.ffn = SwiGLUFFN()

    def forward(self, x):
        x = x + self.mla(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TinyMLAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])
        self.final_norm = RMSNorm(HIDDEN_DIM)
        self.lm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)

    def forward(self, x):
        h = self.embeddings(x)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        return self.lm_head(h)

# ============================================================================
# 5. 训练主循环
# ============================================================================
def train():
    print(f"🚀 Training TinyMLA Language Model on: {DEVICE}")
    model = TinyMLAModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,} ({total_params * 4 / 1024:.1f} KB)")

    start_t = time.time()
    for step in range(STEPS + 1):
        x, y = get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=ByteTokenizer.PAD)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            elapsed = time.time() - start_t
            print(f"Step {step:4d}/{STEPS:4d} | Loss: {loss.item():.4f} | Elapsed: {elapsed:.2f}s")

    print(f"\n🎉 Training complete in {time.time() - start_t:.2f}s!")
    return model

# ============================================================================
# 6. 导出为 FlowServe 二进制 FLSV 格式 (供 WeightLoader mmap 载入)
# ============================================================================
def export_flsv(model: TinyMLAModel, out_path: str):
    print(f"📦 Exporting trained weights to: {out_path}...")
    model.eval().cpu()

    # 1. 组装头部
    MAGIC = 0x464C5356  # "FLSV"
    VERSION = 1
    RESERVED = b"\x00" * 12

    # 2. 顺序平铺参数
    all_floats = []

    # (a) Token Embeddings [VOCAB_SIZE, HIDDEN_DIM]
    all_floats.extend(model.embeddings.weight.detach().numpy().flatten().tolist())

    # (b) 各层参数
    for layer in model.layers:
        all_floats.extend(layer.norm1.weight.detach().numpy().flatten().tolist())
        all_floats.extend(layer.mla.w_dkv.weight.detach().numpy().flatten().tolist())
        all_floats.extend(layer.mla.w_dq.weight.detach().numpy().flatten().tolist())
        all_floats.extend(layer.mla.w_uk.weight.detach().numpy().flatten().tolist())
        all_floats.extend(layer.mla.w_uv.weight.detach().numpy().flatten().tolist())
        all_floats.extend(layer.mla.w_uq.weight.detach().numpy().flatten().tolist())
        all_floats.extend(layer.mla.w_o.weight.detach().numpy().flatten().tolist())
        all_floats.extend(layer.norm2.weight.detach().numpy().flatten().tolist())
        all_floats.extend(layer.ffn.w_gate.weight.detach().numpy().flatten().tolist())
        all_floats.extend(layer.ffn.w_up.weight.detach().numpy().flatten().tolist())
        all_floats.extend(layer.ffn.w_down.weight.detach().numpy().flatten().tolist())

    # (c) Final Norm & LM Head
    all_floats.extend(model.final_norm.weight.detach().numpy().flatten().tolist())
    all_floats.extend(model.lm_head.weight.detach().numpy().flatten().tolist())

    TOTAL_PARAMS = len(all_floats)

    header_fmt = "<IIIIIIIIIQQ12s"
    WEIGHTS_OFFSET = struct.calcsize(header_fmt)

    header_bytes = struct.pack(
        header_fmt,
        MAGIC,
        VERSION,
        NUM_LAYERS,
        HIDDEN_DIM,
        NUM_HEADS,
        HEAD_DIM,
        MLA_LATENT_DIM,
        INTERMEDIATE_DIM,
        VOCAB_SIZE,
        WEIGHTS_OFFSET,
        TOTAL_PARAMS,
        RESERVED,
    )

    with open(out_path, "wb") as f:
        f.write(header_bytes)
        floats_bytes = struct.pack(f"<{TOTAL_PARAMS}f", *all_floats)
        f.write(floats_bytes)

    print(f"✅ Successfully exported {len(all_floats)} parameters to {out_path} ({len(header_bytes) + len(floats_bytes)} bytes)")

if __name__ == "__main__":
    trained_model = train()
    export_path = "/home/caixuf/code/flowserve/tinymla_story.bin"
    export_flsv(trained_model, export_path)
