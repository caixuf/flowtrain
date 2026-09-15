# flowtrain

数据并行 / 张量并行 / 流水线并行的**独立训练调度仓**。和 [flowserve](https://github.com/caixuf/flowserve) 一样，不往 [flowcoro](https://github.com/caixuf/flowcoro) 里塞。

借鉴：

- Megatron-LM：DP allreduce、列切 TP、1F1B
- 细胞仓：先 compile 再 run、定点 MAC、**位级对账**；不是把细胞图拿来当 Transformer

进程组是进程内模拟，不是 NCCL。假线性层，不是真训 LLM。

## 证什么

| 切分 | 对照 | 本机 |
|---|---|---|
| DP | 分 shard 的 `dW` 按 rank 0..W-1 相加后 SGD，权重与单进程参考 **逐 float 相等** | `bit_id=1 vs_ref=1` |
| TP | 输出维切开，`allgather(Y_r)` vs 完整 `X@W` | `bit_id=1` |
| PP | 1F1B vs GPipe：**峰值未回传激活** 10 vs 32。Tf=Tb 且 M 不大时 **不保证** 1F1B makespan 更短 | span 236 vs 226 |

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/bench_train
```

## 不是什么

- 不是 DeepSpeed/Megatron 的完整实现，没有 ZeRO-3、没有 fused Adam、没有真 GPU
- 不是 `flowserve` 的反向传播模式
- IO / `flowcoro` 接入是下一刀

## License

MIT
