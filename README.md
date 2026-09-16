# flowtrain

独立 **训练并行机制**测试台：DP / TP / MLA-TP / 1F1B。不往 [flowcoro](https://github.com/caixuf/flowcoro) 里塞代码。

**学怎么工作：** [docs/LEARN.md](docs/LEARN.md)（定点 GEMM → DP 位级 → compile-then-run 的 1F1B → 有界队列异步流水线）。

借鉴 Megatron 的切分形状和细胞仓的「先 compile 再跑时钟、加法顺序钉死」。  
进程组是**进程内模拟**，不是 NCCL。C++ 测的是假线性层/小维 GEMM，不是用这套并行去训千亿模型。

配套的 `train_tinymla.py` 是**另一条线**：单卡 PyTorch、~10 万参数、导出 FLSV 给 [flowserve](https://github.com/caixuf/flowserve)。

## 证什么

| 切分 | 对照 | 本机 |
|---|---|---|
| DP | shard 的 `dW` 按 rank 顺序相加后 SGD，与单进程参考逐 float 相等 | `test_dp_bit_parity` |
| TP | `allgather(Y_r)` vs 完整 `X@W` | `test_tp_gemm` |
| TP (MLA) | `W_uv` 列切 + `W_o` 行切 vs 单卡，逐 float 相等 | `test_tp_mla` |
| PP 纸面 | 1F1B vs GPipe 活激活 10 vs 32；makespan **不保证**更短 | `test_pp_1f1b` span 236 vs 226 |
| PP 异步 | 协程+容量 2 通道；前向 PTX GEMM 与 CPU 参考位级相等；`gpu_launches=64/64` | `test_async_pipeline` |

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench_train
```

无 NVIDIA Driver / 仅 stub 时，`require_gpu=true` 的异步流水线应直接失败，不要静默改走 CPU。

## 实现要点（细节在 LEARN）

- 前向设备 GEMM：PTX `mul.rn.f32`+`add.rn.f32`，避开 FMA 舍入差。
- 反向 `dW`/`dX` 仍在 CPU；发射计数只锁前向。
- 一 Stage 一 OS 线程 + `wait_task`；通道满时空协程 `yield`，唤醒走 `schedule_coroutine_enhanced`。
- PTX `.target sm_75`，本机 5060 靠 Driver JIT，不是 Blackwell 手写特化。
- TinyMLA：Train PPL≈1、OOD PPL≈35 → 背诵，不是泛化。权重 `tinymla_chat.bin` / `tinymla_story.bin` 给 flowserve mmap。

## 不是什么

没有 ZeRO、FSDP、NCCL、断点、工业数据管线。不要用本仓「硬刚 1B 分布式训练」。对话展示用 Qwen Instruct，见 flowserve Web `--backend qwen`。

## License

MIT
