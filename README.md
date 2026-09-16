# flowtrain

数据并行 / 张量并行 / 流水线并行的**独立训练调度仓**。和 [flowserve](https://github.com/caixuf/flowserve) 一样，不往 [flowcoro](https://github.com/caixuf/flowcoro) 里塞。

借鉴：

- Megatron-LM：DP allreduce、列切 TP、1F1B
- 细胞仓：先 compile 再 run、定点 MAC、**位级对账**；不是把细胞图拿来当 Transformer

进程组是进程内模拟，不是 NCCL。假线性层，不是真训 LLM。

异步 1F1B（`async_pipeline.hpp`）已接入 flowcoro 协程运行时：跨 Stage 用 `BoundedChannel`（容量 2）进行流量控制与反压，前向激活挂载设备端原生 PTX GEMM Kernel 与 CUDA 异步 DMA，通道满/空走 `co_await flowcoro::yield()` 协程级让出调度。

## 证什么

| 切分 | 对照 | 本机 |
|---|---|---|
| DP | 分 shard 的 `dW` 按 rank 0..W-1 相加后 SGD，权重与单进程参考 **逐 float 相等** | `bit_id=1 vs_ref=1` |
| TP | 输出维切开，`allgather(Y_r)` vs 完整 `X@W` | `bit_id=1` |
| TP (MLA) | DeepSeek 潜空间注意力：W_uv 列切多头 + W_o 行切归约，与单进程参考 **逐 float 相等** | `bit_id=1 local_heads=1` |
| PP | 1F1B vs GPipe：**峰值未回传激活** 10 vs 32。Tf=Tb 且 M 不大时 **不保证** 1F1B makespan 更短 | span 236 vs 226 |
| PP (async) | 协程 1F1B 的 dW 按 mb 顺序归约后 vs 单卡参考 **逐 float 相等**；全 Stage 独立 Stream 异步 DMA 往返无损；设备端 PTX GEMM 硬件真实发射且输出逐 bit 全等；活激活 1F1B < GPipe；队列深度 ≤ 2 | `test_async_pipeline`<br>`dma_in_bit=1 gpu_gemm_bit=1 grads_bit=1 gpu_launches=64/64` |

## 构建与测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench_train
```

## 技术实现与严谨边界

### 已实现并闭环验证的
1. **设备端前向 GEMM Kernel**：内嵌通用 PTX 9.0 汇编内核，通过 CUDA Driver API `cuModuleLoadData` + `cuLaunchKernel` 动态 JIT 装载并在 GPU 核心上真实发射执行。严格采用分步 `mul.rn.f32` + `add.rn.f32` 单步舍入，避开硬件 FMA 单次舍入漂移，使设备端计算输出读回后与单核 CPU GEMM 参考保持 **逐 float 位级全等**（`gpu_gemm_bit=1`）。
2. **门禁锁死与发射计数**：`require_gpu=true` 严格禁止静默降级到 CPU；通过硬件发射计数器核准每个 Stage、每个 microbatch 的前向均在 GPU 物理核心上发射（实测 `gpu_launches=64/64`）。
3. **全 Stage 独立 Stream 与 DMA 覆盖**：每个 Stage 独占专属 `CudaStream`，挂载 `DeviceBuffer` + `PinnedHostBuffer` 异步 DMA 往返，彻底解除全局互斥锁，规避驱动多线程 HostFunc 饿死问题。
4. **协程级协作式反压**：通道满/空时采用 `co_await flowcoro::yield()` 挂起让出，经 `schedule_coroutine_enhanced` 派发调度，彻底消灭 `std::this_thread::yield()` 与线程自旋。
5. **梯度保序归约**：每条 microbatch 的 `dW` 独立落地，再按 `mb_id` 0..M-1 严格保序归约，端到端反向梯度与单卡参考 **逐 float 位级全等**（`grads_bit=1`）。

### 明确未做项与架构边界（实事求是，绝不夸大）
- **反向计算仍在 CPU**：设备端 Kernel 目前**仅覆盖前向 $x \cdot W$**；反向梯度传播（$dW = X^T \cdot dY$ 与 $dX = dY \cdot W^T$）目前仍为主机 CPU 的 IEEE-754 定点 GEMM 计算。
- **并发调度模型**：外层调度仍为「一 Stage 一硬件线程 + wait_task」（防止流水线多 Stage 物理线程饥饿）；消灭的是跨 Stage 队列通道满/空时的 OS 线程上下文切换，并非纯单线程驱动全流水线。
- **指令集架构**：PTX 汇编内核指定 `.target sm_75` 通用指令集架构（Turing+），在 RTX 5060 (Blackwell) 上通过 CUDA Driver JIT 动态装载执行，**绝非为 Blackwell 架构专门手写特化微架构指令**。
- **分布式与生产特性**：进程组仍为单机多线程模拟，未接入 NCCL 分布式网络；不是 DeepSpeed/Megatron-LM 等生产级框架，没有 ZeRO-3、混合精度流水与 fused Adam。
- 不是 `flowserve` 的反向传播模式。

## 对话模型与泛化评测 (TinyMLA Chatbot)

> **定位声明**：**框架瘸在「当真训」。**  
> 若把它当成千亿级工业训练栈，它当然不具备 NCCL、ZeRO-3、工业级数据管线与分布式容错；但回到其作为**调度机理微测试台 (Micro-Testbed)** 的真实定位，它是验证自研 Paged KV、Continuous Batching 与 1F1B 协程流控的确定性基准。

配套的 `train_chatbot.py` 将语料严格拆分为训练集与独立泛化测试集（OOD 语义同义改写与未见句子），在 RTX 5060 上运行验证：

| 指标 | 数值 | 说明 |
|---|---|---|
| 训练集规模 | 40 对话对 | 核心常识、FlowServe/MLA 概念与打招呼 |
| 泛化评估集 (OOD) | 8 对话对 | 训练集外同义改写句，独立计算困惑度 |
| 训练步数 / 耗时 | 2000 steps / 12.96s | 本地 RTX 5060 GPU 快速预训练 |
| Train Loss / PPL | 0.0396 / **1.04** | 核心对话对充分拟合 |
| Eval Loss / PPL (OOD) | 3.5538 / **34.94** | 独立泛化测试集困惑度实测 |
| 导出格式 | `tinymla_chat.bin` (404 KB) | 直接由 `flowserve` 经 mmap 零拷贝加载运行流式生成 |

## License

MIT
