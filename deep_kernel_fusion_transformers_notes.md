# Deep Kernel Fusion for Transformers 论文解读

论文：Deep Kernel Fusion for Transformers  
作者：Zixi Zhang, Zhiwen Mo, Yiren Zhao, Robert Mullins  
本地文件：`paper/deep_kernel_fusion_transformers_2602.11808.pdf`  
arXiv：`2602.11808`

## 1. 一句话总结

这篇论文提出 `DeepFusionKernel`，目标是在大模型自回归解码阶段，把 Transformer 中 SwiGLU MLP/FFN 的多段计算深度融合成一个 GPU kernel，尽量避免中间激活写回 HBM 再读回，从而降低内存带宽压力并提升端到端推理吞吐。

它不是改模型结构，也不是减少数学计算量，而是优化同一组 FFN 公式在 GPU 上的执行方式。

## 2. 研究背景

### 2.1 Agentic LLM 推理的瓶颈变化

论文关注的是 agentic LLM inference，也就是带长上下文、长输出、多轮工具调用或代码库处理的推理场景。

这类负载有几个特点：

- 输入上下文和生成输出都更长。
- KV cache 持续增长，占用大量显存。
- batch size 往往受显存容量限制，不能无限增大。
- GPU Tensor Core 算力很强，但数据搬运跟不上。

因此，瓶颈从单纯的计算能力逐渐转向：

```text
HBM 容量
HBM 带宽
片上缓存复用
kernel launch 和中间 tensor 读写
```

也就是说，GPU 不是不会算，而是很多时间花在从 HBM 读权重、写中间激活、再读中间激活上。

### 2.2 为什么关注 FFN/SwiGLU，而不是 attention

近几年很多推理优化集中在 attention，例如 FlashAttention、PagedAttention、FlashInfer 等。但论文指出，在自回归 decoding 中，SwiGLU MLP/FFN 同样是关键瓶颈。

原因是现代 LLM 的 FFN 通常采用 SwiGLU 结构：

```text
A_gate = X W_gate
A_up   = X W_up
A_silu = SiLU(A_gate)
A_2    = A_up * A_silu
Y      = A_2 W_down
```

其中：

```text
W_gate, W_up:   [d_model, d_ffn]
W_down:         [d_ffn, d_model]
d_ffn:          通常是 d_model 的 3.5 到 4 倍
```

这些矩阵非常大，占据模型参数和显存访问的主要部分。尤其在小 batch decoding 时，每个 token 需要读大量 FFN 权重，但可复用的数据量有限，所以容易 memory-bound。

## 3. 原始实现的问题

### 3.1 普通 PyTorch 实现的问题

朴素实现通常会把 FFN 拆成多个算子：

```text
1. X @ W_gate
2. X @ W_up
3. SiLU(gate)
4. up * silu(gate)
5. mid @ W_down
```

这些操作可能对应多个 GEMM kernel 和 elementwise kernel。问题是中间结果会反复经过 HBM：

```text
gate 写 HBM
up 写 HBM
gate 读回做 SiLU
up 和 activated gate 读回做 multiply
mid 写 HBM
mid 再读回做 down projection
```

这会带来两个直接开销：

- 多次 kernel launch。
- 大量中间激活的 HBM load/store。

在 memory-bound decoding 场景下，这些数据搬运会成为主要瓶颈。

### 3.2 两 kernel fused 实现仍然不够深

论文把现有 SGLang/vLLM 一类实现概括为两 kernel 设计。典型形式是：

```text
Kernel 1:
X @ W_gate
X @ W_up
SiLU + multiply
写出 mid

Kernel 2:
mid @ W_down
写出 output
```

相比 PyTorch，这已经融合了 gate/up/activation/multiply，减少了一部分中间读写和 kernel launch。

但它仍然有一个核心问题：

```text
mid = A_up * SiLU(A_gate)
```

这个 `mid` 仍然需要写回 HBM，然后第二个 kernel 再从 HBM 读回来做 `W_down`。当 `d_ffn` 很大时，`mid` 的体积也很大，这一步会产生明显的带宽开销。

### 3.3 为什么不能简单靠更快 GEMM 解决

在大 batch 训练或 prefilling 中，GEMM 的矩阵更大，Tensor Core 利用率更高。但在 decoding 中，token 数较少，batch 受 KV cache 和服务延迟约束，GEMM 形状容易变成小 M、大 K/N 的形式。

结果是：

- 权重矩阵很大。
- 输入 token 数少。
- 权重复用不足。
- HBM 带宽压力大。
- Tensor Core 可能吃不满。

因此，仅优化单个 GEMM 的峰值算力不够，需要减少整个 FFN 路径的数据搬运。

## 4. 论文的核心方法

### 4.1 DeepFusionKernel 的目标

`DeepFusionKernel` 的目标是把 SwiGLU MLP 的多个阶段深度融合：

```text
X
 -> X W_gate
 -> X W_up
 -> SiLU(gate)
 -> up * SiLU(gate)
 -> W_down
 -> Y
```

尽量在一个 fused CUDA kernel 中完成，让中间值通过寄存器、shared memory 或 tile-local buffer 在片上流动，避免 materialize 到 HBM。

论文强调它不增加 FLOPs，优化的是：

```text
减少 HBM traffic
提高 cache reuse
减少 kernel launch
提高 Tensor Core 有效利用率
```

### 4.2 为什么 fusion 适合 FFN

论文区分了两类操作：

- 适合融合：GEMM 后的 elementwise、tile-local 计算、简单逐元素非线性。
- 不适合深度跨阶段融合：需要全局 reduction 或长距离依赖的操作，例如 softmax。

SwiGLU FFN 的中间部分：

```text
SiLU(gate)
up * SiLU(gate)
```

是逐元素计算，非常适合和 GEMM 融合。相比 attention softmax，它没有复杂的跨 token 全局归约依赖。

### 4.3 两阶段数学结构

论文把 MLP 写成两段：

```text
A_2 = (X W_up) * SiLU(X W_gate)
Y   = A_2 W_down
```

传统两 kernel 实现会明确生成 `A_2`。Deep fusion 的关键是不要把 `A_2` 当作完整 tensor 写到 HBM，而是在 tile 级别产生、消费并累加到最终输出。

直观执行方式是：

```text
for token tile:
  for expert/ffn tile:
    计算 gate tile
    计算 up tile
    计算 mid tile = up * SiLU(gate)
    立刻用于 down projection 的局部累加
  写出 output tile
```

实际 kernel 需要处理 tiling、寄存器压力、shared memory 容量、Tensor Core fragment 形状和 occupancy 的平衡。

### 4.4 Tiling 和 loop ordering

论文讨论了两类 tiling 取向。

#### Row-major tiling

row-major tiling 更强调输入激活 `X` 的局部性。

优点：

- 同一批 token 的输入行可以被复用。
- 对 batch 较大，或者 activation traffic 占比较高的场景更有利。

适用倾向：

```text
batch 较大
token tile 内输入复用明显
activation 读写占比较高
```

#### Column-major tiling

column-major tiling 更强调权重 tile 的复用。

优点：

- 更好复用 `W_gate/W_up/W_down` 的局部 tile。
- 对小 batch、大模型的 decoding 更有利。

适用倾向：

```text
batch 较小
模型很大
权重读 HBM 是主要瓶颈
agentic long-context decoding
```

这点对 MoE 也重要，因为 MoE 每个 expert 的 token 数可能很少，单个 expert 的 batch 更小，权重复用更困难。

### 4.5 Profile-driven kernel scheduler

论文没有假设一种 kernel 配置适用于所有模型和硬件，而是引入轻量 profiler-driven scheduler。

流程是：

```text
1. 准备一组候选 fused kernel 配置
2. 推理开始前在目标硬件上快速 benchmark
3. 根据模型 shape、batch size、GPU 架构选择最快版本
4. 后续推理使用选中的 kernel
5. 配合 CUDA Graphs 捕获，避免推理时重复调度开销
```

这样做的原因是最佳配置受多种因素影响：

- `d_model`
- `d_ffn`
- batch size
- GPU 架构，例如 A100/H100
- shared memory 和 register 资源
- Tensor Core 支持
- tensor parallel 的通信拓扑

### 4.6 Tensor Parallel 下的通信处理

论文附录讨论了分布式推理中的 tensor parallel。

对普通矩阵乘：

```text
Y = X W
```

如果拆分权重矩阵，可能需要 all-gather 或 all-reduce 来组合结果。

对 SwiGLU MLP，论文采用的切分思路是：

```text
W_up, W_gate: column-wise partition
W_down:       row-wise partition
```

这样中间的 SwiGLU 计算可以留在本 GPU 上完成，只在 `W_down` 后对最终输出做一次 all-reduce。

结果是：

```text
每个 SwiGLU MLP block 只需要一次 all-reduce
```

这很关键，因为 deep fusion 如果引入额外通信，端到端收益会被通信抵消。

## 5. 实验设计

论文把 `DeepFusionKernel` 集成到 SGLang 推理栈中，而不是只做孤立 microbenchmark。

实验配置：

- 模型：Llama 3.1 70B
- 精度：FP16
- 并行：TP=4
- GPU：4 张 A100 80GB SXM 或 4 张 H100 80GB SXM
- 推理框架：SGLang
- attention backend：FlashInfer
- 启用 CUDA Graphs
- baseline：
  - naive distributed PyTorch
  - SGLang default kernels
  - vLLM
- decoding 测试：
  - prompt length = 1
  - output length = 1024
  - batch size = 1 到 64
- long-generation 测试：
  - output length = 1024、4096、16384
  - batch size = 1、4、16

## 6. 实验结果

### 6.1 相比 SGLang 的端到端吞吐提升

论文报告 `DeepFusionKernel` 集成进 SGLang 后，在 Llama 3.1 70B decoding 上相对默认 SGLang 有额外提升：

```text
A100: 最高 +9.7%
H100: 最高 +13.2%
```

这不是相对 PyTorch 的提升，而是在 SGLang 已经高度优化的基础上继续提升。

### 6.2 batch size 对收益的影响

A100 上收益在小 batch 或 memory-bound 更明显。论文表 1 中 A100 的典型结果：

```text
batch=1:  +5.7%
batch=2:  +7.7%
batch=16: +9.7%
batch=64: +1.3%
```

batch 很大时，GEMM 的计算利用率提高，memory-bound 程度下降，因此 fusion 的边际收益降低。

H100 上因为算力增长更快，内存带宽仍是重要限制，所以在更宽 batch 范围内仍能看到收益。表 1 中 H100 最高在 batch=2 达到 +13.2%。

### 6.3 长输出场景

在 output length 从 1024 增加到 16384 时，attention 和 KV cache 压力会上升，但论文观察到 FFN 仍然占据 per-token latency 的重要部分。

因此，减少 SwiGLU MLP 的 HBM traffic 仍然能带来持续收益。

### 6.4 波动来源

论文指出，吞吐标准差随 batch size 增大而上升，主要原因是 inter-GPU communication jitter。

`DeepFusionKernel` 复用了 SGLang 原有 all-reduce 和 collective primitives，没有改变通信模式，所以通信波动仍会继承 baseline 的特征。

## 7. 和已有工作的区别

### 7.1 相比浅层融合

Apex、TensorRT-LLM、DeepSpeed-MII 等系统中已有一些浅层融合，例如：

```text
GEMM + activation
elementwise pattern fusion
```

但论文认为这类融合没有充分覆盖完整 SwiGLU 路径，尤其没有消除大型中间激活 `A_2` 的 HBM 往返。

### 7.2 相比编译器自动融合

TVM、Welder、Blockbuster 等工作尝试自动 operator fusion。

论文对它们的判断是：

- Welder 依赖 tile-graph cost model，但主要适合线性链。
- TVM 使用 pattern matching 和 heuristic，模板方法更适合较小计算树。
- Blockbuster 展示过 SwiGLU 原型，但偏 standalone compiler study，缺少 runtime feedback 和硬件感知 tuning。

`DeepFusionKernel` 的区别是：

```text
手写/专门设计完整 SwiGLU deep fusion kernel
+ 运行前 profiling scheduler
+ 集成真实推理框架 SGLang
+ 在端到端 decoding 中评估
```

## 8. 论文的主要创新点

### 8.1 把优化重点放在 FFN memory traffic

当前很多推理优化默认 attention 是主战场。论文强调，在 long-context agentic decoding 中，SwiGLU MLP 的大权重和中间激活同样会成为瓶颈。

### 8.2 深度融合完整 SwiGLU 路径

它的目标不是只融合 `GEMM + activation`，而是尽量把：

```text
gate GEMM
up GEMM
SiLU
multiply
down GEMM
```

放进一个深度融合 kernel 中。

### 8.3 减少中间激活 materialization

核心收益来自避免：

```text
mid 写 HBM
mid 从 HBM 读回
```

这一步在两 kernel fused 实现里仍然存在。

### 8.4 按 workload 和硬件选择 kernel

通过 profiler-driven scheduler，在不同 batch、shape、GPU 上选择不同 tiling/loop ordering，而不是依赖固定配置。

### 8.5 保持 TP 通信模式可控

通过合理切分 `W_up/W_gate/W_down`，让中间计算留在本地，只在最后做一次 all-reduce，避免 deep fusion 增加通信成本。

## 9. 局限性

论文自己提到的限制主要是：

- 没有穷尽测试不同 GPU cluster interconnect。
- 没有系统量化 inter-GPU communication 对性能波动的影响。
- 实验主要围绕 Llama 3.1 70B、FP16、TP=4、A100/H100。

从工程角度还可以补充几点：

- 单 kernel deep fusion 对 register/shared memory 压力很高。
- 对不同 `d_model/d_ffn`、batch、expert token 数的泛化需要大量调参。
- 如果融合方式导致重复计算 gate/up，可能得不偿失。
- 对 MoE 场景，expert token 分布不均会让调度和 tiling 更复杂。

## 10. 对当前 MoE FFN kernel 的启发

当前 `custom_fusedmoe.py` 中的 MoE expert FFN 大致是两阶段 fused：

```text
Kernel 1:
X @ W_gate
X @ W_up
SiLU(gate)
up * SiLU(gate)
写出 up_logits

Kernel 2:
up_logits @ W_down
乘 routed expert weight
写出 expert output
```

这和论文里提到的 SGLang/vLLM 两 kernel 设计接近。它已经不是普通 PyTorch 逐算子实现，但还不是论文意义上的 deep fused kernel。

当前实现仍然有：

```text
up_logits 写 HBM
up_logits 从 HBM 读回
```

如果要向论文方法靠近，优化方向是：

```text
在 tile 级别生成 mid = up * SiLU(gate)
立刻用于 down projection 的局部累加
避免完整 up_logits tensor 落 HBM
```

但实现难点是：

- `d_expert` 很大，完整 mid 放不进片上存储。
- `down` GEMM 需要沿 `d_expert` 维度 reduction。
- 如果按 `d_hidden` tile 启动 kernel，可能重复计算 `gate/up`。
- MoE 每个 expert 的 token 数不均匀，部分 expert 的 tile 利用率低。
- 还需要处理 routing、expert weight、scatter/reduce 的额外开销。

因此，当前两 kernel fused 实现是合理的中间版本；论文的 deep fusion 是进一步减少 HBM traffic 的目标形态，但需要更复杂的 tiling、accumulation 和调度策略。

## 11. 关键结论

这篇论文的核心判断是：

```text
在长上下文、大模型、小 batch decoding 中，
FFN/SwiGLU 的性能瓶颈很大程度来自 HBM traffic，
不是 FLOPs 本身。
```

因此，优化重点应该从“单个 GEMM 算得更快”扩展到“整个 FFN 路径少搬数据”。

`DeepFusionKernel` 通过深度融合 SwiGLU MLP、减少中间激活落 HBM、配合 profile-driven scheduler，在真实 SGLang 推理栈里实现了 A100 最高 9.7%、H100 最高 13.2% 的端到端 decoding 吞吐提升。

