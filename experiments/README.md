# 可运行实验

## CliffWalking 官方转移核对

安装 `gymnasium==1.4.0` 后运行：

```bash
python3 experiments/cliffwalking_transition_audit.py
```

脚本逐项比较旧研究所用的确定性转移规则与官方 `CliffWalking-v1` 的转移表。核对范围是从起点可达的非终止状态及其全部动作；它不复核旧研究的训练循环或统计数字。[官方环境说明](https://gymnasium.farama.org/environments/toy_text/cliff_walking/)。

## FrozenLake 等步预算配对探索

安装 `gymnasium==1.4.0`、`numpy==2.3.5` 后运行：

```bash
python3 experiments/frozenlake_pair_pilot.py
python3 experiments/summarize_frozenlake.py
```

本次参数和判据在运行 100 个 cohort 前固定如下：官方 `FrozenLake-v1` 4×4 滑移地图、每回合最多 100 环境步；Q-learning α=0.5、γ=0.99、ε=0.2；每个 cohort 从独立随机初始 Q 表生成 32 个候选，每人恰好准备 1000 次环境交互。初始 Q 元素独立服从 N(0,0.001²)。用官方转移表精确计算贪心策略在 100 步内到达目标的概率，按质量降序取第 8 位强者和第 24 位弱者。只有强者质量比弱者高超过 0.05 且双方初始质量均低于 0.5 时入组。

续训前双方环境状态重置至起点，Q 表保持不变。双方使用同一组逐环境步随机数，续训规则完全相同；每 50 步计算一次精确贪心成功概率，首次达到 0.5 记为首达，最多 3000 步。未首达记为删失，限制时间取 3000 步。主描述量为入组率、初始质量、强−弱限制平均首达差、配对标准误、弱者先达比例及双方删失数。逐 cohort 原始行保存为 CSV，分析脚本据此重算汇总。这个筛选器利用已知转移模型，属于探索性 oracle 对照；不作为未知环境中的实用选组方法，也不把单一地图的 cohort 数当作独立任务数。[官方环境说明](https://gymnasium.farama.org/main/environments/toy_text/frozen_lake/)。

## FrozenLake 官方回合选组

下一批次用 200 个独立官方环境回合的成功率选组，不使用精确转移概率做选择；精确值仅作选组后审计和结果测量。完整参数、入组条件和分析顺序见[冻结协议](../research/2026-11-25_frozenlake_rollout_protocol.md)。运行：

```bash
python3 experiments/frozenlake_rollout_selection.py
python3 experiments/summarize_frozenlake_rollout.py
```

## CliffWalking 等步预算基线

先运行官方转移导出脚本，再运行 Node.js 实验。完整批次协议见[固定记录](../research/2026-11-25_cliffwalking_protocol.md)。

```bash
python3 experiments/export_cliffwalking_transitions.py
node experiments/cliffwalking_equal_steps.js
python3 experiments/summarize_cliffwalking.py
```

## CliffWalking 固定动作优势复核

用准备期 Q 表上的共享参考路径动作优势，分别筛选正向配对和反向对照。阈值与分析顺序见[固定协议](../research/2026-11-25_cliffwalking_margin_protocol.md)。

```bash
node experiments/cliffwalking_margin.js
python3 experiments/summarize_cliffwalking_margin.py
```
