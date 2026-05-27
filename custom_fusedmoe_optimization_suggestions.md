# custom_fusedmoe.py 优化建议

目标文件：`custom_fusedmoe.py`  
目标：优化当前 TileLang MoE FFN kernel，不讨论无关模块或模型结构改造。  

## 1. 当前实现定位

当前 `custom_fusedmoe.py` 实现的是两阶段 fused MoE FFN kernel。

第一阶段：

```text
input @ W_gate
input @ W_up
SiLU(gate)
up * SiLU(gate)
写出 up_logits
```

第二阶段：

```text
up_logits @ W_down
乘 routed_expert_weights
写出 output
```

对应代码结构：

```python
# Step 1: Compute gate and up logits
with T.Kernel(M, T.ceildiv(dexpert, block_dexpert), threads=threads) as (bx, by):
    ...

# Step 2: Compute down logits
with T.Kernel(M, T.ceildiv(dhidden, block_dhidden), threads=threads) as (bx, by):
    ...
```

它比普通 PyTorch 逐算子实现更 fused，但还不是论文 `Deep Kernel Fusion for Transformers` 中的 deep fused kernel，因为中间结果 `up_logits` 仍然会写入 HBM，再被第二个 kernel 读回。

## 2. 主要瓶颈候选

当前实现最可能的性能瓶颈包括：

```text
1. up_logits 写 HBM + 再读 HBM
2. group padding 导致空 block 或低效 block
3. expert token 分布不均导致 tile 利用率低
4. block_token/block_dhidden/block_dexpert 固定，不能适配不同 shape
5. Step 1 中 input 被不同 dexpert tile 重复加载
6. Step 2 对 up_logits 的读取 layout 可能不是最优
7. 如果强行 one-kernel deep fusion，可能出现 register spill 和 occupancy 下降
```

优化时不要默认认为“一个 kernel 一定更快”。实际判断标准应该是：

```text
省掉的 HBM traffic
>
新增的重复计算 + register spill + occupancy 下降 + 更复杂访存
```

如果这个不等式不成立，两阶段 fused kernel 反而更快。

## 3. 优先级最高：增加 kernel variant autotuning

### 问题

当前参数是固定默认值：

```python
block_token=128
block_dhidden=128
block_dexpert=128
threads=256
num_stages=1
```

MoE 的 expert token 数通常高度不均。很多 expert 的 token 数可能远小于 `128`，此时 `block_token=128` 会导致大量空行计算和资源浪费。

### 建议

先建立一组候选 kernel variant：

```text
block_token:    16, 32, 64, 128
block_dhidden:  64, 128
block_dexpert:  64, 128, 256
threads:        128, 256
num_stages:     1, 2, 3
```

对每组配置运行相同输入 shape 的 benchmark，选择最快版本。

### 判断指标

至少记录：

```text
kernel time
DRAM throughput
achieved occupancy
registers per thread
local memory load/store
Tensor Core utilization
```

### 依据

`Deep Kernel Fusion for Transformers` 使用 profiler-driven scheduler，原因是不同模型 shape、batch size、GPU 架构和内存行为下，最优 kernel 配置不同。

MoE 场景更需要这种策略，因为 expert token 分布是动态的。

## 4. 优先级最高：优化 block schedule，减少空 block

### 问题

当前 kernel 中：

```python
M = math.ceil(group_sum / block_token) + group_count
```

同时外部还构造了 `group_padded_offsets`。这会导致一些 block 实际没有有效 token：

```python
actual_rows = T.max(
    0,
    T.min(block_token, cur_group_size - (m_start_padded - group_padded_offsets[cur_group_idx])),
)
```

即使 `actual_rows == 0`，kernel grid 仍然会被启动。由于 grid 还有 `by` 维度，空 block 会被放大：

```text
Step 1 空 block 数 * ceil(dexpert / block_dexpert)
Step 2 空 block 数 * ceil(dhidden / block_dhidden)
```

### 建议

把 padded offset 推导改成显式 block descriptor。

每个有效 block 只记录：

```text
expert_id
real_m_start
actual_rows
```

然后 kernel 中根据 `bx` 直接读取 descriptor：

```text
cur_group_idx = block_expert_ids[bx]
m_start       = block_m_starts[bx]
actual_rows   = block_actual_rows[bx]
```

这样 `M` 应该等于真实有效 tile 数，而不是 padded 后的估算值。

### 预期收益

```text
减少空 block
减少无效 GEMM
减少 group padding 带来的 launch/grid 浪费
简化 kernel 内部索引逻辑
```

### 相关方向

MegaBlocks 和 CUTLASS grouped scheduler 的核心思想都是调度真实有效 tiles，而不是让 padding 决定执行量。

## 5. 优先级高：按 expert token 数分桶

### 问题

MoE 每个 expert 的 token 数不同。固定 `block_token=128` 对大 expert 可能可以，对小 expert 可能浪费。

例如：

```text
expert A: 7 tokens
expert B: 43 tokens
expert C: 180 tokens
```

如果统一用 `block_token=128`：

```text
expert A 约 94.5% 行为空
expert B 约 66.4% 行为空
expert C 需要 2 个 block，第二个 block 仍有 padding
```

### 建议

按 `group_sizes` 分桶：

```text
small expert:  group_size <= 32    -> block_token=32
medium expert: group_size <= 64    -> block_token=64
large expert:  group_size > 64     -> block_token=128
```

可以先用多个 kernel variant 分别处理不同 bucket。

### 风险

多个 bucket 意味着可能启动多个 kernel，kernel launch 增多。需要比较：

```text
减少的空算
vs
增加的 kernel launch 和调度开销
```

在 decoding 小 batch 场景，bucket 数不要太多，先用 2 到 3 桶即可。

## 6. 优先级高：调整 `num_stages`

### 问题

当前默认：

```python
num_stages=1
```

这可能没有充分 overlap global memory load 和 compute。

### 建议

测试：

```text
num_stages = 1, 2, 3
```

分别观察：

```text
kernel time
shared memory usage
occupancy
DRAM stall
```

### 风险

更高 `num_stages` 会增加 shared memory 使用，可能降低 occupancy。它不是单调收益，需要实测。

## 7. 中优先级：优化 `up_logits` layout

### 问题

现在 `up_logits` 的逻辑 shape 是：

```python
intermediate_shape = (group_sum, dexpert)
```

Step 1 写：

```python
up_logits[m_start + i, by * block_dexpert + j] = up_logits_local[i, j]
```

Step 2 读：

```python
up_logits[m_start : m_start + block_token,
          k * block_dexpert : (k + 1) * block_dexpert]
```

这是自然 row-major layout，但不一定最适合 Step 2 的 tile 读取。

### 建议

尝试 block-major layout：

```text
[block_id, dexpert_block, token_in_block, dexpert_in_block]
```

或者：

```text
[expert_id, local_block_id, dexpert_block, token_in_block, dexpert_in_block]
```

这样 Step 2 读取一个 mid tile 时地址更集中，减少跨 expert/padding 的不规则访问。

### 风险

```text
索引更复杂
buffer shape 更复杂
可能影响写入 coalescing
```

建议只在完成 block descriptor 优化后再做。

## 8. 中优先级：减少 Step 1 中 input 的重复加载

### 问题

Step 1 的 grid 是：

```text
M x ceil(dexpert / block_dexpert)
```

每个 `dexpert` tile 都会重新加载同一份 input tile：

```python
T.copy(input[m_start : m_start + block_token,
             k * block_dhidden : (k + 1) * block_dhidden],
       input_shared)
```

如果 `dexpert` 很大，`by` 很多，同一 token block 的 input 会被重复读取多次。

### 建议

探索一个 CTA 或一个调度单元处理多个 `dexpert` tile，共享 `input_shared`。

概念上：

```text
load input tile once
for several dexpert tiles:
  load W_gate/W_up tile
  compute gate/up tile
```

### 风险

```text
需要更多 accum fragment
寄存器压力上升
shared memory 压力上升
可能降低 occupancy
```

如果 register spill 明显，不应采用。

## 9. 中优先级：减少 `up_logits` HBM traffic

### 问题

当前两 kernel 实现的主要代价之一是：

```text
up_logits 写 HBM
up_logits 读 HBM
```

完整 deep fusion 可以消除这一步，但风险很高。

### 稳妥方案

先尝试减少或优化 `up_logits` 的存储成本：

```text
1. 确保只写 actual_rows，避免无效 row 写入
2. 尝试更适合 Step 2 的 block-major layout
3. 如果精度允许，测试 FP8/INT8 中间激活存储
```

### 风险

低精度 `up_logits` 会影响数值正确性，需要和 PyTorch reference 对齐测试。

## 10. 高风险：partial deep fusion prototype

### 目标

避免完整 `up_logits` 落 HBM，在 tile 级别生成：

```text
mid_tile = up_tile * SiLU(gate_tile)
```

然后立刻参与：

```text
output_acc += mid_tile @ W_down_tile
```

### 可探索结构

```text
for hidden_tile:
  output_acc = 0
  for dexpert_tile:
    gate_tile = input @ W_gate_tile
    up_tile   = input @ W_up_tile
    mid_tile  = up_tile * SiLU(gate_tile)
    output_acc += mid_tile @ W_down_tile
  write output
```

### 核心风险

如果每个 `hidden_tile` 都重新计算 `gate/up`，重复计算次数约为：

```text
ceil(dhidden / block_dhidden)
```

这可能直接抵消省掉 `up_logits` HBM traffic 的收益。

### 采用条件

只有当 profiling 显示：

```text
up_logits HBM write/read 是主瓶颈
且 partial fusion 没有明显 register spill
且 occupancy 没有大幅下降
```

才值得继续推进。

## 11. 高风险：split-K / partial output accumulation

### 目标

让多个 `dexpert` tile 并行贡献同一个 output tile：

```text
output[token, hidden] += mid[token, k_tile] @ W_down[hidden, k_tile]
```

### 方案

```text
方案 A: 单个 block 内遍历全部 dexpert，避免 atomic
方案 B: 多个 block split-K，写 partial output，再 reduce
方案 C: 多个 block split-K，直接 atomic_add 到 output
```

### 风险

```text
方案 A parallelism 可能不足
方案 B 引入 partial buffer，又增加 HBM traffic
方案 C atomic contention 和非确定性风险较高
```

这类方案应放在 autotuning 和 block schedule 优化之后。

## 12. 可选：把 expert weight 和 scatter/reduce 融合

当前 `custom_fusedmoe.py` 的 Step 2 已经融合了：

```python
output[m_start + i, by * block_dhidden + j] = output_local[i, j] * routed_expert_weights[m_start + i]
```

但完整 MoE forward 中，外部仍可能需要：

```text
expert_output_routed
scatter_reduce
```

如果允许扩展 `custom_fusedmoe.py` 的接口，可以传入原 token index 和 final output，让 Step 2 直接写回最终 token 位置。

概念上：

```text
atomic_add(final_output[token_idx, hidden], weighted_output)
```

### 收益

```text
减少 expert_output_routed 中间 buffer
减少外部 scatter_reduce kernel
减少一次 HBM write/read
```

### 风险

```text
top-k > 1 时同一个 token 有多个 expert 输出，需要累加
atomic add 可能成为瓶颈
数值顺序改变可能导致微小误差
接口会变复杂
```

如果只允许改 `custom_fusedmoe.py` 内核，不改外部调用，这项暂时不做。

## 13. 不建议优先做的方向

### 13.1 一开始就写完整 one-kernel deep fusion

原因：

```text
寄存器压力大
shared memory 压力大
容易 register spill
occupancy 可能下降
可能重复计算 gate/up
性能不一定稳定
```

deep fusion 是目标方向，但不是第一步。

### 13.2 盲目增大 tile size

大 tile 可能提升 Tensor Core 利用率，但也可能：

```text
增加寄存器
增加 shared memory
降低 occupancy
增加小 expert 的 padding 浪费
```

MoE decoding 下，小 tile 经常更实用。

### 13.3 不看 profiler 直接改算法

必须先确定瓶颈来自哪里：

```text
DRAM bandwidth
Tensor Core utilization
launch overhead
padding 空算
register spill
scatter/reduce
```

否则很容易优化错对象。

## 14. 推荐执行顺序

### Step 1: 建立 profiling baseline

记录当前两个 TileLang kernel 的：

```text
运行时间
DRAM throughput
achieved occupancy
registers per thread
local memory load/store
shared memory 使用
Tensor Core utilization
```

同时记录不同配置下的：

```text
group_sizes 分布
有效 token 数
padding token 数
空 block 数
```

### Step 2: 增加 autotuning

先只调：

```text
block_token
block_dhidden
block_dexpert
threads
num_stages
```

不改算法，避免变量太多。

### Step 3: 改 block descriptor schedule

去掉 `M = ceil(group_sum / block_token) + group_count` 这种粗粒度 padded schedule，只启动真实有效 block。

### Step 4: expert token 分桶

按 `group_sizes` 选择不同 `block_token` 或不同 kernel variant。

### Step 5: 优化 `up_logits` layout

测试 block-major layout 是否改善 Step 2 的读取效率。

### Step 6: partial deep fusion prototype

只在小 shape 上验证：

```text
是否减少 HBM traffic
是否出现 register spill
是否明显降低 occupancy
是否真的比两 kernel 快
```

### Step 7: 根据结果决定是否继续 deep fusion

如果 partial deep fusion 不稳定或收益小，保留两 kernel fused 结构，继续优化 schedule 和 tile。

## 15. 参考方向

- Deep Kernel Fusion for Transformers：SwiGLU MLP deep fusion、减少中间激活 HBM traffic、profile-driven scheduler。
- MegaBlocks：MoE block-sparse/grouped scheduling，减少 padding 和 token dropping。
- ScatterMoE：减少 MoE dispatch/padding/copy 开销。
- Tutel/Flex：MoE 动态负载下的 adaptive parallelism。
- EPS-MoE：动态选择 DenseGEMM/GroupGEMM，重视 workload-dependent kernel selection。
- CUTLASS grouped scheduler：多个 GEMM problem 的 grouped scheduling。
- FlashInfer fused MoE / DeepGEMM：FP8 grouped GEMM 和 MoE kernel 实践。

## 16. 核心结论

对 `custom_fusedmoe.py`，最稳妥的优化路线不是立刻追求单 kernel deep fusion，而是：

```text
1. 先减少空算和 padding
2. 再做 kernel 参数 autotuning
3. 再改善 up_logits 的 layout 和 HBM 读写
4. 最后验证 partial/deep fusion 是否真的收益大于代价
```

当前两阶段 fused kernel 是合理 baseline。deep fusion 只有在确认 `up_logits` HBM traffic 是主瓶颈，并且没有严重 register spill/occupancy 下降时，才值得作为主线推进。

