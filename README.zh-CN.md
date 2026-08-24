<p align="center">
  <img src="docs/assets/agentic-active-autoresearch-hero.jpg" alt="Agentic Active AutoResearch 主动学习闭环与分子空间" width="100%">
</p>

# Agentic Active AutoResearch

**一个 GPU-first、可审计、模型供应商中立的 agentic 主动学习框架。**

[English](README.md) · [快速开始](docs/quickstart.md) ·
[GPU 配置](docs/gpu.md) · [架构](docs/architecture.md) · [安全边界](docs/security.md) ·
[支持矩阵](docs/support-matrix.md) · [论文与库](docs/references.md)

Agentic Active AutoResearch 面向池式主动学习。每个外层 acquisition round 内都有真实的顺序内循环：产生受限
训练候选、在当前已标注集的内部切分上训练和验证、根据已完成 trial 反思并继续产生下一候选，
最后选择最佳方案、用全部已标注数据重训，再预测未标注池。它支持普通表格回归/分类，也支持
SMILES + Morgan fingerprint 或 Chemprop v2 的分子性质任务。

项目刻意把 LLM 权限压缩到安全边界内：它可以根据聚合统计提出训练候选和 acquisition 权重，
但不能看到原始数据、执行代码、改变标注/训练预算或直接控制实验。训练候选会被参数白名单、
数值边界和固定算力约束修复；acquisition 建议也要经过严格 JSON 解析、组件白名单、相对
anchor 限幅、归一化和确定性回退。

![Agentic Active AutoResearch 主架构：顺序训练候选内循环与可审计主动学习外循环](docs/assets/agentic-active-autoresearch-architecture.jpg)

## 核心优点

- **正式训练默认走 GPU**：自动选择 CUDA，再选择 Apple MPS；`require_accelerator: true`
  时没有 GPU 会直接失败，不会悄悄退回 CPU。
- **保留原项目的训练候选内循环**：强制 control、逐 trial 产生候选、内部验证、反思、最佳
  方案全标签重训；不是固定配置的重复训练。
- **五条训练路线真实可运行**：从头训练、历史全局最好 checkpoint 续训、上一轮 checkpoint
  续训、课程学习、鲁棒损失都会进入 PyTorch 内循环；第 0 轮因没有历史权重而自动隐藏两条续训路线。
- **CPU 路线不混淆**：Random Forest 仅用于 CI、快速 smoke 和经典基线，产物中会明确标注。
- **逐样本可解释**：每次选择都记录预测值、不确定性、各 acquisition 分数、最终分数和观测值。
- **公平比较**：seed、split、round、batch、epochs、architecture、ensemble、MC 次数、
  optimizer steps 和 inner trial budget 不由 LLM 修改；续训继承的累计计算血缘另外报告。
- **供应商中立**：任何 OpenAI-compatible endpoint 都可以通过环境变量接入；默认完全离线。
- **任务可扩展**：CSV/JSONL/Parquet/合成数据、可注册 loader、回归/分类、表格/Morgan 特征、自定义模型、策略、acquisition、oracle；内置与仅可扩展能力在支持矩阵中明确区分。
- **可恢复可复现**：配置和数据哈希、原子写入、确定性 tie-break、设备记录、resume 校验和独立 HTML 报告。
- **开源安全边界清楚**：不提交 `.env`、密钥、数据、运行结果、个人路径或 raw LLM response；
  本地 checkpoint 使用无 pickle 的 NPZ、JSON contract 和 SHA-256，并留在被忽略的运行目录。

## 安装与 GPU 检查

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
agentic-autoresearch doctor
```

仓库与发行包名为 `agentic-active-autoresearch`，推荐命令是 `agentic-autoresearch`；为兼容
已有用户，Python import 继续使用 `agentic_al`，并保留旧的 `agentic-al` 命令别名。

分子任务：

```bash
python -m pip install -e '.[molecule]'
python -m pip install -e '.[molecule,chemprop]'  # 需要 Chemprop 时
python -m pip install -e '.[parquet]'            # 需要 Parquet 时
```

GPU 演示：

```bash
agentic-autoresearch demo --output-dir outputs/my-gpu-demo
```

结果中应出现 `"accelerator_used": true`，设备为 `cuda` 或 `mps`。报告在
`outputs/my-gpu-demo/report.html`。

仅用于 smoke 的 CPU 基线：

```bash
agentic-autoresearch demo --cpu-baseline --output-dir outputs/cpu-smoke
```

不要把 CPU smoke 结果当成正式 GPU 模型证据。

CUDA、Apple MPS 和 Chemprop 的逐项检查见 [GPU 配置与排错](docs/gpu.md)。

## 运行配置

```bash
agentic-autoresearch validate-config --config configs/gpu_regression.yaml
agentic-autoresearch run --config configs/gpu_regression.yaml
agentic-autoresearch run --config configs/gpu_regression.yaml --resume
```

resume 时配置、数据、内置源码或去标识化 provider/model 运行指纹只要改变，程序就拒绝续跑，
避免把不同实验拼成一条曲线；API key 本身不会保存或进入哈希。

## 配置训练候选内循环

```yaml
inner_loop:
  enabled: true
  candidate_generator: rule_based  # agent.enabled=true 时 auto 可使用 LLM
  trial_budget: 6                   # 有历史权重后：control + 五条核心路线
  candidates_per_step: 1           # 每次先看完结果，再产生下一候选
  validation_fraction: 0.25
  selection_metric: mae
  selection_mode: min
  require_control: true
  candidate_paradigms:
    - retrain_from_scratch
    - finetune_global_best
    - finetune_previous_round
    - curriculum_training
    - robust_loss_training
  tunable_parameters:
    learning_rate: [0.00001, 0.005]
    dropout: [0.0, 0.5]
    weight_decay: [0.0, 0.01]
```

内循环只使用当前已标注集的内部切分选方案，外部 validation 只用于报告，绝不参与候选选择。
acquisition policy 同样只读取已标注集内部 trial 的聚合指标，不读取外部 validation 指标。
`epochs`、batch size、网络深度/宽度、ensemble、MC passes、optimizer steps、workers 和预算
全部锁定。PyTorch 当前真实支持 scratch、按已标注集 inner metric 选择的历史全局最好续训、
上一轮续训、easy-to-hard curriculum，以及回归 Huber robust-loss。历史 checkpoint 只能来自
本次运行已完成的轮次，采用无 pickle NPZ、JSON contract 和 SHA-256 校验；外部 validation
不参与 checkpoint 或候选选择。完整说明见[训练候选内循环](docs/inner-loop.md)。

## 接入自己的表格数据

```yaml
dataset:
  kind: csv
  path: ../data/my_pool.csv
  task: regression
  id_column: sample_id
  generate_id_if_missing: true
  target_column: activity
  feature_columns: [descriptor_a, descriptor_b, family]
  group_column: series_id
  cost_column: experiment_cost
  risk_column: failure_risk
```

将 `kind` 改为 `jsonl` 可读取逐行 JSON，改为 `parquet` 可读取 Parquet；三种格式采用相同的
schema、泄漏、ID 和缺失值检查。数据库或实验系统可以通过 `register_dataset()` 注册可信
loader。CLI 默认把 target 当作隐藏 oracle 来模拟实验。真实实验请在 Python API 中实现
`Oracle.observe()`，并为真正无标签的回归 pool 单独传入带标签且 ID/group 不重叠的
`validation_data`；同一个 ID 的调用应当幂等，防止恢复运行时重复下单或重复实验。0.1.0
尚不宣称完全无标签的外部分类流程。

## 分子任务

- `configs/qm9_torch_gpu.yaml`：Morgan fingerprint + GPU MLP ensemble + MC Dropout。
- `configs/qm9_chemprop_gpu.yaml`：Chemprop v2 D-MPNN GPU 训练；Morgan 空间只用于 diversity。

把有合法使用权的数据放入 `data/qm9.csv`，不要提交 Git。示例需要 `smiles` 和 `lumo`。
若没有 `sample_id` 会自动生成稳定行 ID。请求 `scaffold` 分组且 CSV 中没有该列时，会用
RDKit 生成 Bemis–Murcko scaffold；大数据建议预先计算。任何 `target_range` 都必须结合数据
单位单独核实；配置里的数值只是格式示例。

## 接入 LLM，但不写 key

```bash
export AGENTIC_AL_API_BASE='https://your-provider.example/v1'
export AGENTIC_AL_API_KEY='...'
export AGENTIC_AL_MODEL='your-model-id'
agentic-autoresearch run --config configs/openai_compatible_gpu.yaml
```

发送内容只有 round、pool/labeled 数量、聚合指标、不确定性摘要、允许调节的参数边界与已完成
trial 的汇总结果。不会发送 SMILES、特征行、row ID、标签、路径、密钥、checkpoint 或原始
响应；只记录清洗后的提议和修复结果。非 localhost 必须 HTTPS；provider 失败可审计地回退到
rule-based policy。

## 自定义训练候选插件

仓库自带一个可运行的 GPU 插件示例，演示如何根据已完成 trial 的聚合结果继续产生候选：

```bash
agentic-autoresearch validate-config \
  --config configs/custom_training_candidates_gpu.yaml \
  --plugin examples/custom_training_candidates.py
agentic-autoresearch run \
  --config configs/custom_training_candidates_gpu.yaml \
  --plugin examples/custom_training_candidates.py
```

`--plugin` 也接受已安装的 Python 模块名。插件是拥有当前用户权限的可信代码，不能由 prompt、
上传数据或不可信配置自动选择。模型、acquisition、dataset、oracle 和训练候选的扩展契约见
[自定义组件](docs/custom-components.md)与[支持矩阵](docs/support-matrix.md)。

## 产物与审计

每个 round 都包含：

- `inner_loop/`：内部切分摘要、候选上下文、candidate batch、逐 trial training plan/result、
  reflection、最佳方案和汇总表；
- `strategy.json`：原始权重、修复记录、最终权重、fallback 状态；
- `metrics.json`：验证指标、训练后端、真实设备、ensemble/uncertainty 信息；
- `selection.csv`：逐样本 prediction、uncertainty、组件分数、final score、observed target；
- `commit.json`：完整 round 的提交标志。

根目录还有 redacted config、环境版本、数据/配置哈希、summary、final metrics、state 和无需服务器
即可打开的 HTML 报告。公开 config 会把本地数据和输出路径替换成占位符。

## 科学表述边界

Agentic Active AutoResearch 的高分候选、proxy label、模型预测或 LLM 建议都不等于实验确认。正式报告至少要给出
多 seed 稳定性、匹配预算的 baseline、sample efficiency、预测误差、不确定性质量、cost、
wall-clock 和失败率。关于经典主动学习、MolPAL、Chemprop、BoTorch、ChemCrow 等工作及其与
本项目的关系，见 [docs/references.md](docs/references.md)。

自适应内循环要与相同训练次数的 random search 对照；不能把“多训练几次”误写成 agent 的
决策增益。模型选择与外部评估分离的依据见 Cawley & Talbot (2010)，随机搜索基线依据见
Bergstra & Bengio (2012)，链接均收录在 references。

## 当前状态

`0.1.0` 是研究型 release candidate。GPU MLP 的回归/分类路线和顺序训练候选内循环均已在
Apple MPS 本地贯通，其中回归内循环真实训练并比较了同预算 Huber 鲁棒损失候选。
Chemprop 2.2.3 适配器也已用程序生成的分子完成一次单 epoch MPS 功能冒烟；CUDA、目标环境
Chemprop 安装和真实数据集性能仍需在发布 benchmark 前验证。仓库地址为
<https://github.com/llmir/agentic-active-autoresearch>。

实际运行环境、测试数字和未验证范围见 [release-candidate 验证记录](docs/validation.md)。
仓库发布遵循[维护者发布清单](docs/releasing.md)，不从本机推断个人身份或上传任何密钥。
旧项目中哪些能力已保留、重新设计或尚未迁移，见
[原型迁移审计](docs/migration-from-prototype.md)。
