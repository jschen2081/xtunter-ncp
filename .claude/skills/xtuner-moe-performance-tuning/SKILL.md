---
name: xtuner-moe-performance-tuning
description: >
  Tune XTuner MoE training throughput and memory with controlled EP, compile,
  intra-layer microbatch, recompute, FSDP sync/reshard, loss chunk, grouped GEMM,
  and SonicMoE. Use for 35B-A3B, 41B-A4B/48-layer and similar models, including
  1M/2M global-token proxy tests and 1024-GPU scale-up planning. Enforces exact
  model/token shapes, resource limits, complete-log validation, OOM classification,
  and reproducible handoff reports.
---

# XTuner MoE 性能调优

目标是在不改变模型结构与训练语义的前提下，找到可复现的吞吐/显存最优组合，并留下他人可以重跑的证据。不要把历史最优数字当作新模型的默认答案，也不要为了得到数字而扩大资源占用、缩小目标 batch 或跳过失败日志。

## 1. 先固定实验契约

开始任何训练任务前，先记录并核对：

- 实际模型配置类及结构：层数、层类型、hidden size、Q/KV heads、head dim、GatedDeltaNet 维度、专家数、top-k、shared expert、expert intermediate size、router、vocab、RoPE、MTP、精度。
- 代码分支、HEAD、未提交修改、训练镜像和关键依赖版本。不要只记录分支名。
- GPU 型号与数量、节点数、namespace、charged group、镜像、数据路径和 tokenizer 路径。
- `SEQ_LEN`、`GLOBAL_BATCH_SIZE`、`WORLD_SIZE`、每步 global tokens、每卡序列数、每卡 tokens。
- 用户允许的资源上限与提交方式。计算 Running、Starting、Inqueue、Pending、Stopping 等预留资源，不能只看 Running。未明确上限时先确认，不能把本案例的 128 卡当成永久默认值。

如果这些信息不完整，不要提交大规模任务。特别是 1024 卡正式实验，必须有用户明确要求，不能用小测试的名义直接启动。

### Token 形状必须先固定

比较性能时，以下量必须一致，否则先重新设计实验而不是直接比较吞吐：

```text
global_tokens = GLOBAL_BATCH_SIZE × SEQ_LEN
tokens_per_gpu = global_tokens / WORLD_SIZE
sequences_per_gpu = GLOBAL_BATCH_SIZE / WORLD_SIZE
grad_accumulation_steps = GLOBAL_BATCH_SIZE / (WORLD_SIZE × INTRA_LAYER_MICRO_BATCH)
```

本案例的可比映射是：32 卡 GBS128/256 对应 1M/2M；1024 卡 GBS4096/8192 对应 32M/64M。它们分别保持每卡 4/8 条序列。不能用每卡 128 条序列的历史大 batch 结果代表生产的每卡 4 条形状。

## 2. 保持模型和训练语义

- 不改变层数、专家数、top-k、router、共享专家、hidden/intermediate size、MTP、词表、训练精度或数据标签。
- 区分数据侧 `pack_level` 与 MoE 层内 routed-token pack。`pack_level="none"` 不意味着 MoE pack 被关闭。
- 不要因为存在 pack commit 就假设 EP 一定走 fast-path。EP 下先检查 dispatcher 的 `supports_micro_batch_pack` 和实际调用路径。
- 配置调优不等于修改模型或 kernel。若需要改代码以获得性能，先明确说明并补正确性验证，不能混入普通配置对照。
- SonicMoE 只走当前支持的 no-EP 路径：`ep_size=1`、`dispatcher=None`。EP + SonicMoE 是功能未实现，不是性能为 0。
- SonicMoE 默认 general routing 保留固定 top-k，不使用 token rounding 改变路由语义。

## 3. 推荐调参顺序

每一步尽量只改变一个主旋钮，但最后必须做联合复核，因为 `ILMB × recompute × reshard × no_sync × loss chunk` 会共同改变显存和通信。

### A. 先验证正确性与内存边界

1. 用小规模/短步验证 forward、backward、loss、grad norm、专家梯度和累计 tokens。若修改了 pack 或 kernel，必须先证明数值与未优化实现一致。
2. 记录首次编译、Triton autotune 与稳定训练步的区别。首步慢不能直接当成稳定性能。
3. 先从 ILMB1/2 确认可运行，再向 ILMB4/8 搜索。不要为了省梯度累积而直接假设大 microbatch 更快。

### B. 先比较主要结构旋钮

在 grouped GEMM、TP=SP=Expert TP=1 下，先比较：

- EP1 与 EP4，保持 dispatcher 语义正确；EP>1 使用 all2all，EP1 使用 None。
- eager 与 compile。EP 路径尤其需要检查 compile，不能用 eager 的结果排除某个 EP 档位。
- ILMB、loss chunk。chunk 不是越大越好，必须同时看 allocated peak、reserved peak、慢步和吞吐。

如果更关心显存，再比较 no-EP SonicMoE，但先解决依赖和编译问题，不要先花大量资源扩展次要 EP 档位。

### C. 联合调同步、reshard 与 recompute

- `no_sync=true` 在多个 microbatch 时可能减少同步，但非最终 microbatch 会保留更多参数/梯度状态。必须与 reshard、ILMB、chunk 联合验证，不能假设它必然更快或必然 OOM。
- `grad_accumulation_steps=1` 时 no_sync 配置值不改变实际同步行为，不要为了这个字段再开无意义的对照。
- `reshard_after_forward=false` 可能减少通信，也可能提高显存甚至变慢，必须实测。
- recompute 层数为 `int(num_hidden_layers × recompute_ratio)`，最后一层按实现不 recompute。降低 recompute 会增加激活驻留，可能引发 OOM 或显存边界慢步；增加 recompute 也不保证总吞吐更高。
- `GBS/(WORLD_SIZE×ILMB)` 必须为整数，并与日志中的 `grad_accumulation_steps` 一致。

### D. 最后检查后端与生产映射

- 在 grouped GEMM 接近最优后，再比较 SonicMoE 的吞吐与显存收益。
- 保持 4 卡 EP group 在 8 卡节点内。不要随总卡数简单把 EP group 扩大到 32/128/1024。
- 1024 卡实验单独验证跨节点 FSDP/EP 通信扩展性，不能把 32 卡吞吐直接线性外推为生产吞吐。

## 4. 测量与验收

### 统一窗口

- 搜索阶段推荐 12 步，排除前 3 步，统计第 4–12 步；最终复核建议至少 20 步并排除前 5 步。
- 四个节点使用本地 rank0 的步耗时，即全局 rank 0/8/16/24，不要让其他 rank 的日志覆盖样本。
- 同时给稳定步耗时均值和中位数。慢步必须保留在原始数据中，不得挑单步最快值。
- 对早期 4 步或单步结果明确标注，不能与 12 步稳定窗口等价比较。

### 吞吐与显存

```text
tokens/s/GPU = global_tokens / WORLD_SIZE / mean_step_time
global_tokens/s = tokens/s/GPU × WORLD_SIZE
```

显存同时记录 allocated peak 和 reserved peak，主表优先使用日志中的 allocated `max_memory`，不要把 reserved 当成 allocated。

### 完整日志校验

每个有效 run 必须验证：

- 每个节点都完成预期步数，而不是只有 rank0 或 node0 完成。
- `text_tokens`、`seqlen_tokens` 与预期每卡 tokens 一致。
- `approximate_total_consumed_tokens == step × global_tokens`。
- loss、grad norm 为有限值，梯度累积次数与配置一致。
- 日志中的 EP、dispatcher、backend、compile、ILMB、chunk、recompute、reshard、no_sync 与提交值一致。
- RJob 状态与训练日志同时检查。训练后退出异常可以保留逐步性能数据，但必须单独标注，不能写成完全成功。

如果存在 `work_dirs/qwen3p5_40b_a3b_speedtest/parse_speedtest.py`，可复用该脚本完成这些校验；没有该文件时，按上述断言自行解析，不能把缺少校验工具当成通过。

## 5. OOM 与失败处理

- OOM 表示该配置在该硬件与实现下失败，不填写 0 tokens/s，也不伪造吞吐。
- 缺依赖、编译 BrokenPipe、停止、未生成训练日志、训练后 SIGSEGV 分开记录，不能全部写成 OOM。
- 停止或排队任务没有有效结论，不要计入有效性能，也不要仅因其停止就断言 OOM。
- 被用户明确要求停止时，使用准确的 job name 和 namespace，并在停止后核对 active/pending 数。未经授权不要随意取消别人的任务。

在保持目标 global tokens 不变时，推荐按以下顺序回退并逐项记录：

1. ILMB 8→4→2→1，降低单 microbatch 激活与临时 buffer。
2. recompute 0.75→0.8→0.85，以更多重算换显存。
3. 保持 reshard=true，并将多 microbatch 的 no_sync 回退为 false。
4. 减小 loss chunk，降低大词表输出层临时显存，但不要假设它必然解决 MoE backward OOM。
5. 再比较 EP1/EP4 的权重驻留与通信权衡。
6. 若 grouped GEMM 显存仍受限，可在 no-EP 下评估 SonicMoE，先验证依赖与正确性。

如果缩减 global tokens 才能运行，必须明确标注形状已改变，不能声称与原 1M/2M 目标等价。

## 6. 41B-A4B、48 层的迁移规则

当前历史案例只有 35B-A3B/40 层，不能假设目标 41B-A4B 的结构已经存在或与其相同。仓库中另一个 48 层模型也不是目标模型，不能混用配置。

先确认目标模型的实际配置类和 HF config，再应用以下规则：

- 48 层 recompute 映射：0.725→34、0.75→36、0.775→37、0.8→38、0.85→40 层。不要继续写“0.75=30 层”。
- 48/40=1.2 只表示层数增加 20%，不代表吞吐必然下降 20%。A4B/A3B 激活参数、实际 GEMM 形状、专家实现、通信和显存边界都可能变化，必须重新测量。
- 可继承的起步语义是 TP=SP=Expert TP=1、grouped GEMM+compile、EP1/EP4 对照、reshard=true、先 no_sync=false，以及小步验证后 ILMB4 的候选配置；不是继承 35B 的吞吐数字。
- chunk 可从 EP1=2048、EP4=1024/4096 起做对照；这些只是搜索起点，不是 41B 固定答案。
- SonicMoE 只在 no-EP 下作为显存优先候选。使用已验证镜像，若 Quack 编译子进程 BrokenPipe，尝试 `QUACK_COMPILE_WORKERS=1`，不要用缺包或编译失败的 run 计算性能。
- 先 32 卡形状验证，再做必要的中间规模通信验证，最后才是用户明确要求的 1024 卡试点。保持每卡 4/8 条序列对应 32M/64M，不改变实验含义。

## 7. 交付标准

每次调优结束至少交付：

1. 最优配置表：global tokens、GBS、EP/dispatcher、backend、compile、ILMB/GA、chunk、recompute、reshard、no_sync、tokens/s/GPU、全局 tokens/s、步耗时、峰值显存。
2. 调参轨迹：哪些旋钮改变了、为什么保留或淘汰，避免只给最终数字。
3. 失败/OOM 表：完整配置、失败类型、已完成步数、原始日志。停止项单独列出或按用户要求排除，不能冒充有效结果。
4. 复现信息：代码提交、镜像、依赖、环境变量、资源限制、稳定窗口、解析命令、run_dir 和 summary。
5. 对目标模型的明确限制：未验证的结构、未测试的 EP 档位、未验证的跨规模通信，不得包装成已知事实。

历史参考交接文档：`work_dirs/qwen3p5_40b_a3b_speedtest/qwen35_35b_a3b_tuning_handoff.md`。该文档中的数值用于复核本次 35B-A3B 实验，不应直接成为 41B-A4B 的吞吐预期。
