# 怎么学 FlowTrain

目标：看懂 **DP 为什么能和单卡逐 float 相等、TP 怎么切、1F1B 为什么省激活、异步流水线怎么用有界队列反压**。  
不是：用本仓从零训 1B，也不是 NCCL 多机。

两套东西不要混：

| 路径 | 是什么 |
|---|---|
| `include/flowtrain/*.hpp` + `tests/test_*` | 并行**机制**测试台（进程内模拟） |
| `train_tinymla.py` / `train_chatbot.py` | 单卡 PyTorch 训 ~10 万参数 TinyMLA，导出 FLSV |

C++ 测过的切分，并没有用来训那个 PyTorch 小模型。

## 0. 先跑门禁

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench_train
```

`test_async_pipeline` 成功时应看到类似：

`dma_in_bit=1 gpu_gemm_bit=1 grads_bit=1 gpu_launches=64/64`

64 = 4 stage × 16 microbatch 的**前向** kernel 次数。反向仍在 CPU。

## 1. 定点 GEMM：对账的前提

读 `include/flowtrain/tensor.hpp` 的 `gemm`：固定 i-j-k，浮点 MAC 顺序确定。  
多卡参考实现必须按**同一加法顺序**累加，否则 IEEE 下「位级相等」不成立。这是细胞仓那套戒律，不是 GPU 更快。

## 2. DP：分数据，加梯度

读 `include/flowtrain/dp.hpp` + `tests/test_dp_bit_parity.cpp`。

每个 rank 一份 shard → 本地 `dW` → 按 rank 0..W-1 **排好序再加** → SGD。  
和单进程扫完全部数据后的权重比 `==`，不是 `atol`。

自己改：把归约改成乱序累加，门禁应红。这就能体会「位级」有多脆。

## 3. TP：切权重，对齐完整 GEMM

读 `tp.hpp`、`tp_mla.hpp`。列切输出维、行切再 AllReduce。MLA 版：`W_UV` 列切头、`W_O` 行切，潜空间投影各 rank 全量。  
参考实现必须按相同 rank 顺序求和，才能 `forward_bit_identical`。

## 4. PP：先 compile 再跑时钟

这是本仓最值得抄的结构。读 `include/flowtrain/pipeline.hpp`：

1. `compile_gpipe`：每 stage 先全部 F，再全部 B。
2. `compile_1f1b`：warmup `P-rank-1` 个 F，然后 (F,B) 成对，最后 cooldown B。
3. `simulate_pipeline` 是**唯一时钟**：按依赖选下一个就绪 op，统计 makespan 和活激活。

结论要分开记：

- 活激活：1F1B < GPipe（本机 compile 模拟约 10 vs 32）。
- makespan：Tf≈Tb 且 M 不大时，**不保证** 1F1B 更短（文档里 236 vs 226）。省的是内存不是必然墙钟。

练习：把 `compile_1f1b` 的 warmup 写错，`simulate` 应死锁断言。

## 5. 异步 1F1B：从纸面表到真队列

读 `include/flowtrain/async_pipeline.hpp`。

- 每 stage 一条 `std::thread` + `Task`（不是单线程协程泵整条流水线）。
- 跨 stage：`flowcoro::BoundedChannel` 容量 2。满/空 `co_await yield()`，唤醒必须走 `schedule_coroutine_enhanced`。
- **最后一级 F 的输出不要推进容量 2 的队列**，GPipe 会先做完所有 F 再 B，队列塞满就自己等自己。输出放在 stage 本地，B 再读。
- 前向：H2D → PTX `vector_matrix_gemm`（`mul.rn`+`add.rn`，避开 FMA 一次舍入）→ D2H，与 CPU `gemm` 逐 float 比。
- `dW` 按 microbatch 先存，再按 `mb_id` 0..M-1 折起来，才能和串行参考位级相等（并发完成顺序 ≠ 加法顺序）。

## 6. TinyMLA 脚本（另一条线）

`train_tinymla.py` / `train_chatbot.py`：单卡、hidden=64、语料极少。  
chat 的 Train PPL≈1、OOD PPL≈35 表示**背得熟、泛化无**。不要用它证明「训练框架能训对话模型」。

## 建议阅读顺序

`tensor.hpp` → `dp.hpp` → `tp.hpp` / `tp_mla.hpp` → `pipeline.hpp` → `test_pp_1f1b.cpp` → `async_pipeline.hpp` → `tests/test_async_pipeline.cpp`
