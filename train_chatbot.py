#!/usr/bin/env python3
"""
FlowTrain: TinyMLA 对话聊天机器人 (Chatbot) 本地 GPU 预训练与微调脚本
基于 DeepSeek Multi-Head Latent Attention (MLA) 架构，在 NVIDIA RTX 5060 上训练，
并导出为 FlowServe 兼容的 FLSV 二进制模型格式。
"""

import math
import os
import struct
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. 结构与超参数配置 (与 flowserve 完全对齐)
# ============================================================================
VOCAB_SIZE = 259       # 3 特殊标记 (BOS=0, EOS=1, PAD=2) + 256 单字节
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 4
HEAD_DIM = 16
MLA_LATENT_DIM = 16    # 4x 压缩 (4*16 -> 16)
INTERMEDIATE_DIM = 128
MAX_SEQ_LEN = 128
BATCH_SIZE = 32
STEPS = 2000
LR = 2.5e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 2. Byte-level 分词器
# ============================================================================
class ByteTokenizer:
    BOS = 0
    EOS = 1
    PAD = 2
    BYTE_OFFSET = 3

    def encode(self, text: str, add_bos=True, add_eos=True) -> list[int]:
        tokens = [self.BOS] if add_bos else []
        tokens.extend([b + self.BYTE_OFFSET for b in text.encode("utf-8")])
        if add_eos:
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
# 3. 对话语料库 (中英文混合智能聊天与问答)
# ============================================================================
DIALOGUES = [
    # 1. 打招呼与问候
    ("User: Hello!\nAssistant:", " Hello! I am TinyMLA, an AI chatbot running on FlowServe. How can I assist you today?"),
    ("User: Hi\nAssistant:", " Hi there! Nice to meet you. What can I do for you today?"),
    ("User: Good morning\nAssistant:", " Good morning! Hope you have a wonderful and productive day ahead."),
    ("User: Good evening\nAssistant:", " Good evening! How was your day? I am here if you need to chat."),
    ("User: How are you?\nAssistant:", " I am doing great and running at full speed on your RTX 5060! How are you?"),
    ("User: 你好\nAssistant:", " 你好！我是基于 FlowServe 架构运行的 TinyMLA 聊天助手，很高兴为你服务！"),
    ("User: 你好呀\nAssistant:", " 嗨！很高兴见到你，今天想聊点什么呢？"),
    ("User: 早安\nAssistant:", " 早上好！祝你今天心情愉快，工作顺利！"),
    ("User: 晚上好\nAssistant:", " 晚上好！今天过得怎么样？有什么我可以帮你的吗？"),

    # 2. 天气与日常闲聊 (新增天气对话支持)
    ("User: 今天天气不错呀\nAssistant:", " 是呀！今天天气晴朗明媚，微风正好，祝你一天都拥有美丽好心情！"),
    ("User: 今天天气怎么样？\nAssistant:", " 今天阳光明媚，温度宜人，非常适合出门走走散步！"),
    ("User: 天气真好\nAssistant:", " 好天气总能带来好心情，愿你今天也充满活力！"),
    ("User: How is the weather today?\nAssistant:", " It looks bright and sunny outside! A great day to enjoy the outdoors."),
    ("User: 吃饭了吗？\nAssistant:", " 哈哈，我作为运行在 GPU 上的 AI 不需要吃饭，但算力电量随时拉满！你吃了吗？"),
    ("User: 在干嘛呢？\nAssistant:", " 我正在 FlowServe 引擎里高速待命，准备回答你的任何奇思妙想！"),
    ("User: 辛苦啦\nAssistant:", " 能为你效劳一点也不辛苦，随时陪你畅聊！"),
    ("User: 哈哈\nAssistant:", " 哈哈！笑一笑十年少，有什么开心的事情跟我分享吗？"),

    # 3. 身份与自我介绍
    ("User: Who are you?\nAssistant:", " I am TinyMLA, an ultra-lightweight language model trained with FlowTrain and served by FlowServe."),
    ("User: What is your name?\nAssistant:", " My name is TinyMLA, a compact model built with DeepSeek MLA architecture."),
    ("User: Who created you?\nAssistant:", " I was trained with the FlowTrain pipeline on RTX 5060 and served by FlowServe in pure C++20."),
    ("User: 你是谁？\nAssistant:", " 我是 TinyMLA，一个采用 DeepSeek MLA 架构的轻量级语言模型，运行在 FlowServe 引擎上。"),
    ("User: 你的名字叫什么？\nAssistant:", " 我的名字叫 TinyMLA，很高兴认识你！"),
    ("User: 谁训练了你？\nAssistant:", " 我由 FlowTrain 训练套件在本地 RTX 5060 GPU 上预训练完成。"),

    # 4. 功能与能力
    ("User: What can you do?\nAssistant:", " I can chat with you, answer questions about AI architectures, tell stories, and demonstrate ultra-fast C++ token streaming."),
    ("User: 你能做什么？\nAssistant:", " 我可以陪你聊天、回答技术问题、讲童话故事，并演示毫秒级流式推理。"),
    ("User: Can you write code?\nAssistant:", " Yes! I can explain C++ coroutines, CUDA streams, and model architectures."),
    ("User: 可以聊聊吗？\nAssistant:", " 当然可以！天文地理、代码架构、生活趣事，我都乐意与你交流。"),

    # 5. 核心技术常识
    ("User: What is FlowServe?\nAssistant:", " FlowServe is a pure C++20 LLM inference engine with Multi-Head Latent Attention and paged KV cache."),
    ("User: 什么是 FlowServe？\nAssistant:", " FlowServe 是一个自研 C++20 推理引擎，支持 DeepSeek MLA 架构与 mmap 零拷贝权重加载。"),
    ("User: What is MLA?\nAssistant:", " MLA stands for Multi-Head Latent Attention. It compresses key-value states into a low-dimensional latent space to save memory."),
    ("User: 什么是 MLA？\nAssistant:", " MLA 是多头潜空间注意力机制，通过低秩压缩将 KV 缓存缩减 4 倍，大幅节省显存。"),
    ("User: What is FlowCoro?\nAssistant:", " FlowCoro is a high-performance C++20 coroutine runtime designed for heterogeneous compute and async scheduling."),
    ("User: 什么是 1F1B？\nAssistant:", " 1F1B 是流水线并行训练调度策略，交替执行前向与反向，能够显著降低激活显存峰值。"),

    # 6. 趣味问答与数学
    ("User: What is 1 + 1?\nAssistant:", " 1 + 1 equals 2."),
    ("User: 1加1等于几？\nAssistant:", " 1 加 1 等于 2。"),
    ("User: Tell me a joke.\nAssistant:", " Why do programmers prefer dark mode? Because light attracts bugs!"),
    ("User: 讲个笑话\nAssistant:", " 为什么程序员喜欢深色模式？因为亮光会招来 bug！"),
    ("User: Tell me a story.\nAssistant:", " Once upon a time, Lily found a magic glowing flower in the dark forest and made a wish."),
    ("User: 讲个故事\nAssistant:", " 从前有一只勇敢的小兔子，它跳过绿树林，找到了一片甜甜的胡萝卜田。"),

    # 7. 礼貌与道别
    ("User: Thank you!\nAssistant:", " You are very welcome! Let me know if you have any other questions."),
    ("User: 谢谢你\nAssistant:", " 不客气！随时乐意为你效劳，祝你有美好的一天！"),
    ("User: Bye\nAssistant:", " Goodbye! It was a pleasure chatting with you. Have a great day!"),
    ("User: 再见\nAssistant:", " 再见！期待下次与你继续交流。"),
]

def build_training_sample(prompt: str, answer: str):
    full_text = prompt + answer
    tokens = tokenizer.encode(full_text, add_bos=True, add_eos=True)
    if len(tokens) > MAX_SEQ_LEN:
        tokens = tokens[:MAX_SEQ_LEN]
    else:
        tokens = tokens + [ByteTokenizer.PAD] * (MAX_SEQ_LEN - len(tokens))
    return tokens

def get_batch():
    batch_tokens = []
    for _ in range(BATCH_SIZE):
        prompt, answer = DIALOGUES[torch.randint(0, len(DIALOGUES), (1,)).item()]
        batch_tokens.append(build_training_sample(prompt, answer))
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
    print(f"🚀 Training TinyMLA Chatbot on: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"📊 Training Corpus: {len(DIALOGUES)} Dialogue Pairs | Steps: {STEPS} | Batch Size: {BATCH_SIZE}")
    model = TinyMLAModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"📐 Total Parameters: {total_params:,} ({total_params * 4 / 1024:.1f} KB)")

    start_t = time.time()
    for step in range(STEPS + 1):
        x, y = get_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=ByteTokenizer.PAD)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            elapsed = time.time() - start_t
            print(f"  Step {step:4d}/{STEPS:4d} | Loss: {loss.item():.4f} | Speed: {step / max(0.01, elapsed):.1f} steps/s")

    print(f"\n🎉 Training complete in {time.time() - start_t:.2f}s! Final Loss: {loss.item():.4f}")
    return model

# ============================================================================
# 6. 导出为 FlowServe 二进制 FLSV 格式
# ============================================================================
def export_flsv(model: TinyMLAModel, out_path: str):
    print(f"📦 Exporting trained weights to: {out_path}...")
    model.eval().cpu()

    MAGIC = 0x464C5356  # "FLSV"
    VERSION = 1
    RESERVED = b"\x00" * 12

    all_floats = []
    # (a) Token Embeddings
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

    print(f"✅ Exported {len(all_floats)} parameters to {out_path} ({len(header_bytes) + len(floats_bytes)} bytes)")

if __name__ == "__main__":
    trained_model = train()
    export_path = "/home/caixuf/code/flowserve/tinymla_chat.bin"
    export_flsv(trained_model, export_path)
