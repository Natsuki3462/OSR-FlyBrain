# 交接文档 — 果蝇脑驱动 OSR6

> 写于 2026-10-28。当前 PC 停止推进，本文档保证换机器后能无缝继续。

---

## 0. 一句话现状

**跑通了一条完整链路：FlyWire 真实连接组 → 606 神经元 LIF 网络（1 kHz）→ 六轴映射 → TCode → OSR6 六轴模型（3D 可视化）**，闭环含本体感觉回流。

界面是**脑活动主视图 + 右侧 3D 抽屉**，连接手动触发（模拟器 / 真机串口二选一），
未连接时一个字节都不下发。真机入口已接好（`body/device.py` 的 `RealDevice`，带
`D1` 身份校验），但**本机没有串口设备，未在真机上验证**。

但**这台 PC 的算力把方案锁死在"功能子电路"级别**，全脑不可能实时。这是停下的根本原因，不是代码问题。

---

## 1. 本机算力实测（以及 2026-10-29 的优化）

### 本机配置

| 项 | 值 |
|---|---|
| CPU | Intel i5-4210M（2 核 4 线程，2014 年，Haswell） |
| 内存 | 7.9 GB |
| GPU | AMD R7 M265 + Intel HD 4600，**无 NVIDIA，无 CUDA** |

### LIF 网络规模 vs 实时倍率（1 ms 步长，每神经元约 28 突触）

同一台机器、同一把尺子（`tools/bench_step.py --scale`），两种后端：

| 神经元 | 突触 | 纯 numpy | numba JIT | 提升 |
|---:|---:|---:|---:|---:|
| 606 | 16,000 | 17x | **82x** | 4.8x |
| 2,000 | 56,000 | 8.6x | 37x | 4.3x |
| 5,000 | 140,000 | 4.9x | 20x | 4.0x |
| 10,000 | 280,000 | 1.7x | 10x | 6.0x |
| 20,000 | 560,000 | 跑不动 | 4.1x | — |
| 50,000 | 1,400,000 | 跑不动 | 1.7x | — |
| 138,639 | ~5,000,000 | 跑不动 | ~0.5x（外推） | — |

**结论：优化前本机上限 5,000 神经元，现在约 5 万（实时），保守可用 2 万（4 倍余量）。**

注意表里的放电率是刻意压高的合成网络（2 万神经元时 72 个/步）。真实电路静息时
0.4 个/步，传播成本低得多，所以实际能跑的规模比表里还大一些。

全脑 13.8 万神经元在 1 kHz 下仍需约 250 倍于本机的算力；文献常用的 0.1 ms 步长
（10 kHz）还要再乘 10 倍。这条路 CPU 优化救不了，只能上 GPU。

### 2026-10-29 做了什么（5 千 → 5 万）

先剖析再改，四步，每步都验证过：

1. **numba JIT 内核**（`brain/lif_jit.py`）。原来每步要做 12 次 numpy 调用，
   n=606 时 Python 开销和浮点运算各占一半。现在整步压成一个 `@njit` 函数，
   4 趟循环合并成 1 趟（延迟抑制的读取与当前步传播无关，可以提前做）。
   传播改用 CSR：只在阈值循环里收集真正放电的神经元，只遍历它们的出边，
   成本 O(活跃数 × 出度)，与网络总规模无关。
2. **噪声池**。`rng.standard_normal` 每步 13.7 us，占总时间 30%。改成预生成
   一块扁平噪声循环使用——样本独立、顺序无关，统计上等价。
3. **`world.step` 从 34 us 降到 8 us**。`drive[:] = 0` 每步清整条数组，改成
   只清上一步真正置非零的那几十个位置；泊松计数和选择索引改成预生成池
   （用「随机排列分块拼接」保证同一块内取前 k 个互不重复，语义与原
   `choice(replace=False)` 一致）。
4. **映射与输出**。mapper 每帧做 10 次 fancy-index 求均值（150 us），改成
   一次 gather + `np.add.reduceat`（5 us）。`emu.send6` 原来走
   `run_until_complete`，每次多付 50 us 事件循环调度，delay=0 时改走同步路径。
   闭环里的 `np.isin` 换布尔掩码，`active[loom].sum()` 换 `dot` 掩码。

数值等价性：`tools/verify_jit.py` 在关闭噪声、同一 drive 序列下对比两个后端，
**4000 步放电序列逐位一致**，`max|dv| = 1.9e-6`（float32 舍入累积）。

端到端（20000 ms 闭环）：2540 ms → 1080 ms，**7.9x → 18.5x**。

后端选择：默认 `auto`（有 numba 就用）。`OSR_LIF_BACKEND=numpy` 可强制退回，
`LIFNetwork(..., backend="numpy")` 同理。numba 首次运行要编译约 10 秒，之后
走 `__pycache__` 缓存。

### 换机器能解决什么

| 升级方向 | 预期收益 | 备注 |
|---|---|---|
| 现代 CPU（i7-13700 / Ryzen 7 7800X3D） | 2–3 倍（单核 IPC + 频率） | numba 内核是单线程的，核多帮助有限；要吃到多核得按空间切块做区域分解，步与步之间有依赖，改动大 |
| 任意 NVIDIA GPU + CUDA | 10–100 倍 | 需要重写引擎（Brian2CUDA / GeNN / NEST GPU） |
| RTX 4070 及以上 | 可考虑 10 万级神经元，接近全脑 | `eonsystemspbc/fly-brain` 要求 Linux + RTX 4070 |

**换 PC 后第一件事：重跑规模曲线，看新机器的拐点在哪。**

```powershell
python tools\bench_step.py --scale 606 2000 5000 10000 20000 50000
```

表里 20,000 那一行的实时倍率如果 > 3x，就可以直接把子电路扩到 2 万；如果
只有 1.2x，就停在 5 千到 1 万之间。这个数字是唯一依据。

---

## 2. 目录结构与代码地图

```
OSR-FlyBrain/
├── HANDOFF.md                    ← 本文档
├── README.md                     ← 项目说明（面向外部读者）
├── requirements.txt
├── setup.ps1                     ← 一键重建环境（Windows）
├── start.bat                     ← 一键启动（推荐，不受执行策略限制）
├── run.ps1                       ← 同上（需要先放开执行策略）
│
├── brain/                        ← 脑
│   ├── lif.py                    ← 1 kHz LIF 引擎（核心，约 200 行，双后端）
│   ├── lif_jit.py                ← numba 内核（CSR 传播 + 单趟合并循环）
│   └── circuit.py                ← 电路加载、按角色/类型/侧别索引
│
├── body/                         ← 身体
│   ├── emu.py                    ← 原版 osr_emu.py 的同步适配层
│   ├── device.py                 ← RealDevice（TCode 真机）+ LinkManager（三态连接）
│   ├── mapping.py                ← 六轴映射（核心）
│   ├── world.py                  ← 虚拟环境，产生刺激事件
│   ├── server.py                 ← HTTP 服务 + 实时闭环线程
│   ├── osr/                      ← 原项目 osr.py / osr_emu.py 原封不动搬进来
│   │   ├── osr.py                ← 六轴定义、TCode 编码
│   │   └── osr_emu.py            ← 模拟器（异步）
│   └── web/
│       ├── sim.html              ← 前端：脑活动主视图 + 3D 抽屉 + 连接按钮
│       └── vendor/
│           └── osr-emu.dist.min.js  ← 原项目的 3D 模型库（WebGL，8 MB）
│
├── data/
│   ├── circuit.json              ← 抽好的电路（208 KB，关键产物）
│   ├── circuit_571_backup.json   ← 加 jo_down 之前的版本（对照用）
│   └── raw/                      ← FlyWire 原始 CSV（55 MB，可重新下载）
│
├── etl.py                        ← 从原始 CSV 抽子电路 → circuit.json
├── demo_brain.py                 ← 纯脑演示（无身体），打印各群放电率
├── brain_loop.py                 ← 主入口：脑-体闭环
│
├── tests/
│   ├── test_behaviour.py         ← 脑行为不变量（19 项）
│   └── test_body.py              ← 身体层（26 项）
│
└── tools/
    ├── fetch.py                  ← 并行分块下载器（可续传）
    ├── bench_step.py             ← 单步耗时分解 / 规模曲线 / 真实负载
    ├── verify_jit.py             ← numba 与 numpy 后端数值一致性
    ├── profile_loop.py           ← 闭环分段计时
    ├── diagnose.py               ← 查某群为什么不放电
    ├── selectivity.py            ← loom vs 非感觉的触发率对比
    ├── wind_path.py              ← 风感通路逐段贯通性检查
    ├── probe_jo.py               ← JO 在真实连接组里的上下游分布
    ├── shot_page.js              ← CDP 截图（真实时间等待 + Page.captureScreenshot）
    └── cdp_eval.js               ← 在页面里执行 JS 并打印结果（调试运行时状态）
```

---

## 3. 数据来源与电路内容

### 数据源

**FlyWire FAFB v783**（成年雌果蝇，仅脑，不含腹神经索）
- 138,639 神经元 / 约 500 万突触
- 公开下载，无需注册：`https://storage.googleapis.com/flywire-data/codex/data/fafb/783/`
- 四个文件：`classification.csv.gz`、`coordinates.csv.gz`、`consolidated_cell_types.csv.gz`、`connections.csv.gz`

> **注意**：v783 **没有 wind 细胞类型**。风感用的是 `JO-*`（江氏器）系列。
> **注意**：`consolidated_cell_types.csv.gz` 的列名是 `root_id, primary_type, additional_type(s)`，**没有 `cell_type` 列**。

### 当前电路（`data/circuit.json`）

606 神经元 / 8,126 突触。

| 角色 | 细胞类型 | 数量 | 含义 |
|---|---|---:|---|
| `loom_lc4` | LC4 | 104 | 逼近物检测 |
| `loom_lplc2` | LPLC2 | 210 | 逼近物检测 |
| `escape_gf` | DNp01 | 2 | **巨纤维 / 逃逸** |
| `steer_left` | DNa01 | 2 | 左转 |
| `steer_right` | DNa02 | 2 | 右转 |
| `forward` | DNp09 | 2 | 前进 / 步速（气流驱动，见 §4.7） |
| `backward` | MDN | 4 | 后退 |
| `groom` | DNg11 | 6 | 梳理 |
| `wing_a/b/c` | DNp02/04/11 | 2+2+2 | 翅膀 / 起跳 |
| `ascending` | AN | 36 | 上行（本体感觉回流） |
| `wind` | JO-* | 20 | 江氏器（风 / 翅振） |
| `jo_down` | CB 系局部中间神经元 | 35 | JO 第一跳中继（2026-10-29 新增，见 §4.8） |
| `partner` | 各种 | 177 | 中间神经元 |

**独立验证**：11 个核心类型在 v783 中的数量与 DesktopFly 公布值完全一致（LC4=104、LPLC2=210、DNp01=2、DNa01=2、DNa02=2、DNp09=2、DNg11=6、MDN=4、DNp02/04/11 各 2）。这说明抽取方法本身是对的。

---

## 4. 关键设计决策（以及为什么）

### 4.1 为什么是"功能子电路"而不是全脑

见 §1。这是算力决定的，不是设计偏好。

### 4.2 为什么权重用幂次归一化

`etl.py` 的权重公式：

```
w = syn / (total_in^α · ref^(1-α)) · sign(nt)
```

- `α=1.0` 时退化为"每个神经元总输入归一化到固定值"——稳定但**丢失选择性**
- `α=0.0` 时退化为全局缩放——**一触即雪崩**（p90 输入达 7.7 倍阈值）
- 实际取 `α=1.0` + 后续的 `GAIN_E=3.2 / GAIN_I=1.8`

神经递质决定符号：ACh/DA/OA/5HT = 兴奋；GABA/GLUTAMATE/HISTAMINE = 抑制。

### 4.3 为什么必须有 SFA（锋电位频率适应）

没有 SFA 时网络是**双稳态**的：低于阈值完全静默，一旦越过就雪崩，中间没有稳定工作点。

参数：`ADAPT_GAIN = 0.28`，`τ = 200 ms`。

### 4.4 为什么伙伴神经元要"每群私有 + 特异性过滤"

- 取各群 top-N 的**并集** → 这些神经元成了共享枢纽，把所有通道焊死在一起
- 改成**每群私有**（互不重叠）→ 仍然不够
- 再加**特异性过滤**（`SPECIFICITY=0.20`，要求伙伴对该目标群的输出占比 ≥20%）→ 才出现真选择性

### 4.5 为什么刺激必须是 2 ms 同步脉冲

40 ms 恒定驱动会让 loom 神经元**异步乱放**，巨纤维退化成"任何活动都响应"的探测器。

真实 looming 是一次同步波（60 个神经元在 2 ms 内齐射），一致性检测才起作用。改成脉冲后，选择性从 7/20 直接变成 0/20（对照）。

### 4.6 为什么 escape 需要 loom 同步门控

加上行持续驱动后，巨纤维会**自发**放电（约 1 Hz）。原因：上行 → GF 的 11 条边全是抑制性的，但上行激活 partner 网络，partner 有 226 条边喂给 GF。

单看 GF 放电无法区分真逃逸和自发。但**因果耦合**可以：

| 场景 | GF 放电前 8 ms 内 loom 群单步最大放电数 |
|---|---|
| 真 loom 事件 | 50–60 |
| 自发 | 全部为 0 |

阈值原取 5 即可做到 **19/19 通过、0/25 误报**。**2026-10-29 起改成 35**，因为加了持续气流驱动（见 §4.7）后，伙伴网络被长期轻度激活，会顺带把 loom 群顶到单步 19–26 个——虽然远低于真 loom 的 60，但已经超过 5。取 35 后真 loom（60）全过、气流（19–26）全挡，两边各留 25 和 9 的裕度。

### 4.7 L0/L1 的映射，以及前进通路是怎么修好的

**原来的结论是「`forward`（DNp09）在这个子图里任何驱动下都不放电」**——这条是错的，2026-10-29 推翻了。

用 `tools/diagnose.py --role forward` 查出来：DNp09 的兴奋输入 3.20、抑制 1.80、净 +1.40，上游 30 个。**直接驱动这 2 个神经元 → 200 ms 内 80 个尖峰；驱动它上游 20 个同步 → 716 个尖峰。** 通路一直是通的，缺的是闭环里的**同步驱动源**：`world` 只给上行神经元随机泊松驱动（每秒 20 Hz × 36 个，不同步），凑不出 DNp09 需要的同步输入。

修法是给 `world` 加一路**持续气流**（`flow_hz=6`，每次 2 ms 脉冲，驱动 DNp09 的 12 个上游），代表飞行中一直存在的气流，而不是离散事件。三个约束是试出来的：

1. **不能驱动 loom 群的上游**。DNp09 的 20 个上游里有 3 个是 LPLC2，一驱动就在 loom 群内部扩散，同步峰值顶到 26，混进逃逸门控变成假逃逸。
2. **只驱动 ascending 不够**。那 3 个上行神经元即使 amp=5 也点不着 DNp09（实测 0 次），必须带 partner。
3. **驱动 partner 会顺带激活 loom 群**（19 个/单步），因为伙伴网络共享。这条消不掉，只能靠抬高门控阈值挡（见 §4.6）。

同时把 L1 的前进系数从 `0.10` 提到 `0.35`：以前 forward 不放电，给大了也没用；现在给足了 L1 才有真正的双向。

| 轴 | 修前 | 修后 |
|---|---|---|
| L1 范围 | 0.225 → 0.563（只有后退） | **0.231 → 0.834（前进后退都清晰）** |

L0 仍然只靠 `wing`（振翅→升力）和 `escape`（急退），因为 DNp09 到 L0 的贡献本来就小（`0.20 × fwd`），保持原样。

### 4.8 风感通路：JO → jo_down → DNp02（2026-10-29 补）

**原来的结论是「JO 能驱动自身，但下游响应为 0，输出目标不在子图内」**。这条对了一半。

`tools/diagnose.py --role wind` 显示 20 个 JO 神经元在子图里**完全孤立**：上游 0、出边 0。用 `tools/probe_jo.py` 查真实连接组，JO 的出边有 77 条 / 37 个对象，几乎全是 **CB 系列局部中间神经元**；第二跳能命中 `wing_a`（88 突触）、`escape_gf`（51）、`backward`（6）、`wing_c`（5）。所以不是"没有下游"，是**这一层没被拉进子图**。

修法：`etl.py` 加 `[4.5/5]` 阶段，把 JO 的直接下游（35 个）拉进来，角色标 `jo_down`。规模 571 → 606，突触 7,730 → 8,126。用 `--no-jo-down` 可以生成旧电路做对照。

实测（`tools/wind_path.py`，6000 ms）：

| 群 | quiet | wind | 变化 |
|---|---:|---:|---|
| `jo_down` | 0.4 Hz | 0.9 Hz | **2.6x** |
| `wing_a`（DNp02） | 0.1 Hz | 0.4 Hz | **5x** |
| `wing_b` / `wing_c` | 0.3 / 0.2 Hz | 0.3 / 0.2 Hz | 不变 |
| `escape_gf` | 0.3 Hz | 0.3 Hz | **不变**（风不伪装成逃逸） |
| `backward` | 1.4 Hz | 1.4 Hz | 不变 |

`wing_b`/`wing_c` 不动是对的：它们在连接组里本来就没有 JO 输入（`jo_down → wing_b` 0 条、`→ wing_c` 2 条），只有 DNp02 有 8 条。

**两个负面结果，都很重要：**

1. **这条通路只有 1–3% 的权重占比。** `wing_a` 的兴奋输入里 LC4 占 70.8%、partner 27.0%、`jo_down` 只有 2.3%（`#136` 是 3.3%）。单次 2 ms 风脉冲让 25 个 `jo_down` 在 1 ms 内同步放电，但 `wing_a` 只从其中 3 个接到输入、权重合计 **0.073**，离阈值 1.0 差 14 倍。这是连接组的真实比例，不是 bug——JO→AMMC→DNp02 本来就是弱通路。
2. **加增益救不了。** 试了给 `jo_down` 出边乘 1–64 倍：

   | 增益 | R2 均值 | V0 均值 | wind−quiet 的 R2/V0 差 |
   |---:|---:|---:|---:|
   | 1 | 0.606 | 0.224 | +0.034 / +0.072 |
   | 8 | 0.642 | 0.301 | +0.025 / +0.052 |
   | 32 | 0.657 | 0.334 | +0.016 / +0.035 |
   | 64 | 0.657 | 0.334 | ~0 |

   绝对电平随增益上升，但 **quiet 基线一起上升**（`jo_down` 有 148 条自身抑制性循环边 + 60 条上行抑制边，靠噪声维持 0.4 Hz 紧张性放电），风的**特异性**反而被冲掉。所以没有加这个旋钮。

结论：风感通路的缺口补到了连接组允许的极限——**能调制，不能驱动**。

---

### 4.9 前端：脑活动主视图 + 3D 抽屉 + 手动连接（2026-10-29 重构）

界面按 `OSR-Control` 的布局重做：**主区是脑，3D 模型退到右侧抽屉**。

**三态连接（`body/device.py` 的 `LinkManager`）**

| 状态 | `target` | `send6()` 行为 | 界面 |
|---|---|---|---|
| 未连接 | `None` | **直接返回 `(0, "")`**，不构造也不发送任何 TCode | 顶栏「连接大脑模型」，抽屉遮罩 |
| 模拟器 | `emu` | 转给 `EmuDevice`（3D 模型） | 顶栏「模拟器 · SR6」，抽屉显示模型 |
| 真机 | `serial` | 转给 `RealDevice`（TCode over 串口） | 顶栏「真机 · COMx」，抽屉灰度 + frozen |

"按下按钮才连接"就是靠这张表实现的——**未连接时 `send6` 是空操作**，
一个字节都不出去。`/state` 里同时给两个字段：
`tc`（真正下发的，未连接时是空串）和 `tc_preview`（将要下发的预览值），
前端未连接时显示后者并把标签改成「TCODE · 预览（未下发）」。

**真机身份校验**：`RealDevice.connect()` 连上后必须读到 `D1` 含 `"TCode"`，
否则当作连错设备并硬关传输层。必须做——`find_serial_port()` 在"机器上只有
一个串口"时会无条件返回它，很容易连到无关设备。

**脑图投影**：FlyWire voxel 坐标（x≈295k–732k）先居中、再按最大跨度归一化到
`[-1,1]`，前端只做 yaw/pitch 旋转 + 透视缩放。`/brain` 返回 `pos`（归一化）/
`vox`（原始 voxel 整数，浮层显示用）/ `role` / `type` / `side` / `edges`，只发一次。
**坐标别让前端反算**：`pos` 只保留 4 位小数，乘回 `span` 会有 ±23 voxel 的误差，
所以 `vox` 单独发一份整数。

**性能**：8126 条突触画在离屏 canvas 上，兴奋/抑制各一条 path 只 stroke 两次；
静态时每帧只 `drawImage` blit 一次。视角自动旋转时隔帧重建（4.6°/s 下错位 <0.2°），
拖动时每帧重建保证跟手。实测边层 1.3 ms、节点层 0.5 ms、`renderer.render` 2.3 ms，
整体 60 fps。

**踩过的两个坑：**

1. `resize()` 里 `window.dispatchEvent(new Event("resize"))` 会无限递归——`resize`
   本身就挂在 `window.resize` 上。实测帧率掉到 7 fps。改成只在 `fold()` 里发一次。
2. 抽屉折叠时 `#dmask` 没被一起隐藏。抽屉只剩 56 px，遮罩里的中文被挤成一列竖排，
   而且 `z-index` 比 `#strip` 高，把展开按钮整个盖住。`body.collapsed #dmask{display:none}`
   必须加（`.show` 的 `display:grid` 要靠更高特异性压过去）。
3. 抽屉固定 392 px，窗口窄于 1000 px 时主区会被压到一百多 px。`resize()` 里加了一条
   自动收起（`innerWidth < 1000` → `fold(true)`），用户仍可手动展开。
   **但只能做一次**：`fold(false)` 自己会触发一次 `resize`，如果每次都判断，用户刚点开
   抽屉就会在 320 ms 后被重新收回去（实测 900 px 宽时 drawer 330 → 56）。
   现在用 `autoFolded` 标志位保证只自动收一次，之后完全听用户的。
4. 浮层是 **`pointerup` 触发**（`moved < 4` 才算点击），不是 hover。
5. **菜单项必须用事件委托**。原来是 `renderMenu()` 里 `querySelectorAll("[data-act]")`
   逐个绑 `onclick`，而「已连接」那个分支是 `m.innerHTML = html; return;`——**提前 return
   跳过了绑定块**，结果「断开连接」点上去完全没反应。改成在 `#menu` 上委托一次，
   之后 `renderMenu` 怎么重写 `innerHTML` 都不会再丢 handler。菜单里还有动态生成的
   端口列表（`[data-port]`），同一套委托一起处理。

---

### 4.10 捕食者闭环：让「逃逸」真的有用（2026-10-29 补）

以前 `World.step()` 只按时间表发 loom，**从不读六轴**，所以不管果蝇做什么，
威胁都按固定节奏来——「躲了也没用」，这不是闭环，是开环演示。

现在 `demo` 模式里有一个真正的捕食者（`body/world.py`）：

- 相对坐标里果蝇在原点，捕食者在 `(_px, _py)`。捕食者每步以 `PRED_V`
  朝原点逼近，初始在后方 `(-40, +9)`。
- 果蝇的六轴反灌回世界（`world.set_axes(axes, esc)`，在主循环 `mapper.update`
  之后调用，**下一帧**生效——相当于一层 ~20 ms 感觉运动延迟）：
  - `L1` 的偏离 → 巡航平移（死区 0.08 滤掉姿态抖动，后退 ×0.2，因为走比飞慢）
  - `L2` 的偏离 → 转向（`FLY_TURN`），改变前进方向
  - 逃逸强度 `esc` → 一次 **弹道式前冲**（`FLY_V_ESCAPE`），这才是把捕食者
    推开的那一下。**不能只从 L1 反推**：巡航和逃逸的 L1 幅值太接近，
    混在一起就分不出「窜逃」和「晃一下」。
- 距离 < `LOOM_DIST=12` 才 loom，而且是每 15 ms 刷新一次的 2 ms 同步脉冲，
  不是定时器。距离被推开 → loom 自动停 → 逃逸结束（这就是「距离变远就取消
  这次 loom」）。
- 最近距离有下限 `PRED_D_MIN=2.5`（贴脸但不穿模）。
- **出现是偶发的**：一次威胁被躲开（或超过 `PRED_MAX_ACTIVE=10 s` 兜底）后，
  捕食者撤离到 `PRED_GONE=52` 离场，进入等待；**下一次出现 = 上一次出现时刻 +
  U(10, 50) s**（`PRED_GAP_MIN/MAX`），由 `_spawn()` 在进场时排定。等待期间
  不发 loom，`pred_on=False`。
- 状态机：`wait → active（逼近 / loom）→ leave（撤离）→ wait`。

60 s 实测：出现时刻 0 / 36.1 / 55.1 s，间隔 **36.1 s、19.0 s**（都在 10–50 s）；
每次威胁约 3.4 s 后撤离（逃逸把距离推开）。HUD 在等待期显示「无威胁」。

**关键接线**：`brain_loop.py` / `body/server.py` 在 `mapper.update()` 之后必须调
`world.set_axes(axes, mapper.diag["escape"])`。漏了这行，逃逸对世界毫无影响，
又退化成开环。

**模式**：`live`（正式，默认）与 `demo` 都有捕食者；`quiet` / `flow` / `wind` 是
受控实验场景，没有捕食者，行为与旧版一致（测试 §5 的 `sync_max < 35`、§6 的
「风不伪装成逃逸」都依赖这一点）。`World(..., predator=None)` 默认按模式决定，
也可显式传 `True/False`。

---

### 4.11 手动刺激：左侧抽屉 + `/api/stim`（2026-10-29）

UI 左边有一个默认收起的抽屉（浮层，不改 grid 布局），里面 7 个按钮，按下立刻
触发，不用等下一次事件：

| kind | 目标群 | 幅值 / 时长 | 效果 |
|---|---|---|---|
| `loom` | `loom_lc4 + loom_lplc2` | 3.0 / 6 ms | 巨纤维逃逸 |
| `food` | `forward`（DNp09） | 3.2 / 500 ms | 前趋（**无嗅觉通路，代理**） |
| `wind` | `wind`（JO） | 3.0 / 220 ms | 风感 |
| `flow` | `flow`（forward 上游） | 2.2 / 1500 ms | 持续前进 |
| `wing` | `wing_a/b/c` | 2.6 / 600 ms | 振翅 |
| `groom` | `groom`（DNg11） | 2.4 / 1400 ms | 梳理 |
| `backward` | `backward`（MDN） | 3.0 / 500 ms | 后退 |

链路：`POST /api/stim {"kind": ...}` → `SimRunner.push_stim()` 写入
`SimpleQueue` → 仿真线程每毫秒 `get_nowait()` 后 `world.trigger(kind, t)` →
`step()` 里 `_apply_manual()` 把目标群加到 `drive`。队列线程安全，HTTP 线程与
仿真线程不会互相踩。

两个坑：

1. **`DIRTY_CAP` 必须 ≥ n**。手动 `loom` 一次点亮 314 个神经元，原来 256 的
   dirty 表直接 `could not broadcast (314,) into (256,)`。现已改成 640（>606）。
2. **方法不能叫 `stim`**。`SimRunner.__init__` 里有 `self.stim = stim`（刺激规模，
   int），同名方法被实例属性覆盖，调用时报 `'int' object is not callable`。
   改名为 `push_stim`。

隔离测试（静默网络，baseline/noise=0）：7 个刺激都产生对应动作，`loom` 是唯一
触发巨纤维的（GF=6），其余 GF=0——不会误触发逃逸。

---

## 5. 踩过的坑（按重要性排序，全部是真实调试过程）

### 5.1 抑制性突触符号写反 ★最严重

`w_i` 存的是**幅值**（正数），但延迟队列被 `v +=` 加到膜电位上——等于抑制性输入一直在**兴奋**神经元。

这一个 bug 让之前所有调参结果全部作废。修复：

```python
v -= self._dl[self._dl_pos]   # 不是 v +=
```

### 5.2 mapping 的时间常数按 1 ms 写，但实际 20 ms 才调一次

`steer_alpha` 和 `smooth` 用 `1 - exp(-1/tau)` 算，假设 dt=1 ms。但 `update()` 实际每 20 ms 调用一次 → **时间常数被拉长 20 倍**。

表现：转向基线变成 160 秒、输出平滑几乎不动，六轴范围只有 ±0.03，看着像"脑没反应"。

修复：所有 alpha 都用实际 `dt` 算。

### 5.3 `LOOM_WINDOW=8` 小于 20 ms 帧长

8 ms 的环形缓冲装不下整帧。loom 事件若发生在帧的前 8 ms，到帧末已被覆盖成 0 → **escape 永不触发**。

修复：`LOOM_WINDOW=25`（略大于帧长 20）。

### 5.4 测试对照组包含了巨纤维自己

随机对照群从全部 684 个神经元里抽，把 GF 也抽进去了。直接驱动 GF 当然 t=0 就放电。

这个 bug 让我一度得出"完全没有选择性"的错误结论。修复：对照组必须排除 GF。

### 5.5 上行持续驱动会破坏 GF 的选择性

想用"持续本体感觉输入"维持脑活跃，结果 GF 对 loom 的响应从 20/20 掉到 1/20，而自发涨到 4 次。

试过排除投射到 GF 的 2 个上行神经元（521、525），**无效**——partner 网络被集体激活才是主因。

**结论：任何上行持续驱动都会破坏 GF 选择性。** 最终方案是保留上行驱动（为了让轴动起来）+ loom 同步门控（区分真逃逸）。

### 5.6 其他

- **LIF 基准测试一开始给出 2.45x 的假数据**——突触投递循环根本没跑（0 尖峰）。修好激励后真实值是 1.8x。
- **不应期算成 3 ms**——`refr_steps = round(2.0/1.0) = 2`，但倒计时在设置的当步就减了。修成 `round(refract/dt) - 1`。注意：不应期是**下界**不是固定周期，弱驱动下间隔会自动拉长。
- **`pip install numpy` 报 "No matching distribution"**——`pip config` 里的清华源返回 502。加 `--index-url https://pypi.org/simple` 解决（8.2 MB/s）。
- **curl 下载 Google Storage 停滞**——实测 148 KB/s → 0 B/s 反复。写 `tools/fetch.py` 4 路并行分块 `Range` 请求后 324 KB/s。8 路会触发 502，`CHUNK` 必须 256 KB、`timeout` 25 s。
- **curl 访问 localhost 返回 502**——系统代理拦截。必须加 `--noproxy '*'`。
- **`rm -rf` 被沙箱安全机制拦截**（目录里文件数 > 50）。换新目录名，不要删。
- **`.\run.ps1` 报 `UnauthorizedAccess`**——Windows 默认执行策略禁止跑 `.ps1`。
  三条路：①每次 `powershell -ExecutionPolicy Bypass -File .\run.ps1`；②当前会话
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`；③**改用 `start.bat`**（已提供，
  批处理不受该策略限制，这也是默认推荐）。顺带记两个写 `.bat` 的坑：
  - **`.bat` 不要写中文**。文件是 UTF-8 而 cmd 按 GBK 解析，多字节字符会吃掉换行，
    整行被当成命令执行（实测报 `'默认' 不是内部或外部命令`）。要中文就存成 GBK。
  - **`echo` 里不能有裸 `>`**。`echo fly brain -> six axes` 会把 `->` 当成重定向，
    在项目根目录生成一个名为 `OSR6` 的垃圾文件。
- **`python` 在 PATH 上不一定是有 numpy 的那个**。本机 `where python` 第一个命中
  `.workbuddy-ai/binaries/python/versions/3.13.12`（无 numpy），而 `py -3` 指向
  Python 3.14 + numpy 2.5.3。所以启动器要**按候选逐个探测**（`.venv` → `py -3` → `python`），
  不能直接写 `python`。
- **Windows 的 `SO_REUSEADDR` 是"允许抢端口"**，不是"跳过 TIME_WAIT"。`HTTPServer` 默认
  `allow_reuse_address = 1`，于是两个进程能同时 bind 同一个端口，请求随机落到其中一个。
  实测被坑过：改了代码，但浏览器打到了半小时前启动的旧进程，怎么看都是"改动没生效"。
  `body/server.py` 的 `_Server` 已经把 Windows 下的这个开关关掉，端口被占用会直接
  打印「端口已被占用」并退出（退出码 2）。**排查这类"改了没反应"先看有没有僵尸进程**：
  `Get-NetTCPConnection -LocalPort 8099 -State Listen` → `Get-Process -Id <pid>`。

---

## 6. 已知缺口（按优先级）

| 缺口 | 影响 | 修复思路 |
|---|---|---|
| ~~**`forward`（DNp09）不放电**~~ | ~~L1 只能后退~~ | **2026-10-29 已修**，见 §4.7 |
| ~~**JO/wind 通道下游响应为 0**~~ | ~~风感完全无效~~ | **2026-10-29 已补**，见 §4.8。通路通了，但只有 1–3% 权重，能调制不能驱动 |
| **跨通道串扰随刺激规模上升** | 刺激 120 个神经元时对照触发率升到 14/20 | 606 个神经元太少，但**扩大子图会破坏选择性**，见 §6.1 |
| **权重没有生理标定** | 绝对放电率不可信，只有相对响应可信 | 需要真实电生理数据标定；或用 DesktopFly 的公开参数 |
| **三个轴只有噪声驱动** | L2/R0（转向）、R1（梳理）、R2/V0（振翅）在动，但靠上行泊松背景，没有对应感觉输入 | 需要补 DNa/DNg/DNp 的各自感觉上游；JO 已补（§4.8），视觉→转向还没接 |
| **真机入口已接好但未验证** | 只跑了模拟器；`RealDevice` 没在真机上跑过一次 | `body/device.py` 的 `RealDevice` 复用 `body/osr/osr.py`（pyserial 3.5 已装），带 `D1` 含 `TCode` 的身份校验。本机 `/api/ports` 返回 `[]`，插上设备后从顶栏菜单选端口即可 |
| ~~3D 截图工具未验证成功~~ | ~~没法自动截图~~ | **2026-10-29 已跑通**，见 §7.4（关键是别加 `--disable-gpu`） |

### 6.1 扩大子电路会破坏选择性（2026-10-29 实测）

想靠 `etl.py --partners` 扩规模，试了三档，用 `tools/selectivity.py` 同一把尺子量：

| 电路 | 神经元 | 突触 | loom 触发 | 非感觉触发 | 比值 |
|---|---:|---:|---:|---:|---:|
| **当前**（含 `jo_down`） | 606 | 8,126 | **19/20** | 4/20 | **4.8x** ✅ |
| 加 `jo_down` 前（备份 `circuit_571_backup.json`） | 571 | 7,730 | 20/20 | 5/20 | 4.0x ✅ |
| `--partners 150 --specificity 0.08` | 1,144 | 19,429 | **0/20** | 0/20 | — ❌ |
| `--partners 120`（默认 specificity） | 749 | 8,750 | 2/20 | 10/20 | 0.2x ❌ |

**规模一大，loom 就完全不触发巨纤维了。** 原因是权重归一化：`w = syn / tot_post × GAIN_E`，`tot_post` 是**该神经元所有上游的突触总数**。上游从 248 涨到更多时，每个突触的权重按比例缩小，loom 那几个神经元的贡献被稀释，60 个同步齐射再也推不到阈值。

所以 `SPECIFICITY=0.20` 不是保守，是**必要条件**：它保证进来的伙伴是「专门投射到目标群」的，这些才是有效的驱动源。放宽它换来的规模是虚的。

注意加 `jo_down` 这一层没有破坏选择性（19/20 vs 4/20），因为它是**给 JO 补下游**、不是给 loom 那群补上游——`wing_a` 的 loom 占比只从 72% 掉到 70.8%。

### 6.2 归一化到底该怎么改（2026-10-29 量化了）

§6.1 提的替代方案是「按全脑突触总数归一化」：`w = syn / tot_post(全连接组) × GAIN_E`。这样每个边的权重就**与子图里还有谁无关**，扩规模不再稀释。

实测了一下这条路要付多少代价（`exc_full / 子图兴奋边数`，606 电路 322 个有输入神经元）：

| 角色 | 全脑兴奋突触/神经元 | 子图兴奋边/神经元 | 比值 |
|---|---:|---:|---:|
| `loom_lplc2` | 213 | 2.7 | 105 |
| `partner` | 1,410 | 27.8 | **89** |
| `escape_gf` | 2,440 | 229.5 | **10.6** |
| `wing_a` | 1,492 | 110.5 | **13.6** |
| `wing_b` | 1,660 | 166.5 | **10.0** |
| `steer_right` | 9,746 | 53.5 | **182** |

整体中位 **88**，10% 分位 13.8，90% 分位 380，最大 1,536。

**比值不是常数——跨度两个数量级。** 命令神经元采样密（比值 10–14），伙伴神经元采样稀（比值 89），转向神经元最稀（182）。换成全脑归一化，等于把所有相对强度按各自不同的系数重排一遍，`GAIN_E=3.2`、逐群 `REF`、`LOOM_SYNC_MIN`、`flow_*` 全部要重标定。这不是"改一行"，是一个独立课题。**现阶段不做。**

---

## 7. 换机器后怎么继续

### 7.1 环境重建

**前置**：Python 3.11+、Node 18+（可选，只用于截图工具）。

```powershell
# Windows PowerShell，在项目根目录执行
.\setup.ps1
```

或手动：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements.txt
```

> 依赖是 `numpy` + `numba`（numba 可选但强烈建议，快 4-6 倍；没装会自动退回
> numpy，不会报错）。不要用国内镜像源，会 502。
>
> numba 首次运行要 JIT 编译约 10 秒，之后走 `__pycache__` 缓存。如果看到
> 第一次跑测试特别慢，是正常的。

### 7.2 验证是否可用（按顺序）

```powershell
# 1) 跑全部测试（45 项，应该全通过）
.\.venv\Scripts\python.exe tests\test_behaviour.py
.\.venv\Scripts\python.exe tests\test_body.py

# 2) 看脑的行为（无身体）
.\.venv\Scripts\python.exe demo_brain.py --ms 900 --loom-at 300

# 3) 跑闭环，看六轴范围
.\.venv\Scripts\python.exe brain_loop.py --ms 20000

# 4) 开可视化
.\run.ps1
# 浏览器打开 http://127.0.0.1:8099/
```

**关键验证点**：
- 测试 [7] 的实时倍率：本机 **75–82x**（numba）/ 17x（numpy）。如果只有 17x，说明 numba 没装上
- `python -c "from brain.circuit import Circuit; from brain.lif import build_from_circuit; print(build_from_circuit(Circuit()).backend)"` 应打印 `numba`
- `brain_loop.py` 输出里"有动作的轴"应该是 7 个
- "逃逸触发 N 次"应该 > 0

### 7.3 三条路线

**路线 A：扩大子电路（⚠️ 先读 §6.1，直接扩会毁掉选择性）**

本机优化后 606 神经元还有 82 倍余量，2 万神经元仍有 4 倍，所以不用换机器就能
先把电路扩一个量级：

```powershell
# 先确认余量
python tools\bench_step.py --scale 606 5000 20000
# 再按余量调 partners
python etl.py --partners 120 --out data/circuit_big.json
```

`--partners` 当前是 30（606 神经元）。调到 120 约 2,000 神经元，调到 400 约
6,000 神经元。经验法则：**目标规模的实时倍率要留 3 倍以上余量**——mapper /
emu / world 还要吃掉每步约 17 us 的固定开销，而且规模越大放电率越高、
CSR 传播成本也跟着涨。

> 注意：改 `partners` 后**必须重新标定 `body/mapping.py` 里的 `REF` 字典**，因为放电率会整体变化。标定方法见 §7.5。

**路线 B：补感觉通路（2026-10-29 已完成两条）**

- **DNp09 前进**：电路本身没问题，缺的是闭环里的同步驱动源。修法在 `body/world.py` 的持续气流，见 §4.7。
- **JO 风感**：不是"没有下游"，是下游那一层没被拉进子图。修法在 `etl.py` 的 `[4.5/5]` 阶段，见 §4.8。
  通路的权重占比只有 1–3%，能调制不能驱动——这是连接组的事实，加增益反而会冲掉特异性。

下一个同类缺口是**转向 / 梳理 / 振翅没有感觉上游**（L2、R0、R1、R2、V0 目前只被上行泊松背景推动）。
照 §4.8 的套路做：先 `probe` 出 DNa01/DNa02/DNg11/DNp02 的真实上游，再在 `etl.py` 里加对应阶段。

**路线 C：上 GPU**

需要重写引擎。候选：
- `Brian2CUDA` — 生态成熟，但要改写电路描述
- `GeNN` — 性能最好，学习曲线陡
- `eonsystemspbc/fly-brain` — 现成的全脑实现，要求 Linux + RTX 4070，GPL-2.0

### 7.4 关于 3D 截图（调试工具）

**已跑通。** 用 `tools/shot_page.js`：

```bash
# 先单独起一个带 CDP 的 Edge（后台）
"/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
  --headless=new --no-sandbox --hide-scrollbars \
  --remote-debugging-port=9223 --window-size=1440,900 \
  --user-data-dir="<repo>/data/_edgeprofile" about:blank

node tools/shot_page.js "http://127.0.0.1:8099/" out.png 7000 1440 900
node tools/shot_page.js "http://127.0.0.1:8099/" out.png 6000 1440 900 'document.querySelector("#btnLink").click()'
```

第 7 个参数是截图前要在页面里执行的 JS（用来点开菜单、切状态）。

**关键坑：**

1. **不要用 `msedge --headless --screenshot`**。页面里 `requestAnimationFrame` 永不空闲，
   `--virtual-time-budget` 永远结束不了，渲染进程被判定异常终止，拿不到图。
2. **不要加 `--disable-gpu`**。加了之后 WebGL 走 SwiftShader 软件渲染，
   `renderer.render` 单帧从 **2.3 ms 变成 49.8 ms**，帧率掉到 6 fps——
   这是测试环境的假象，不是页面性能问题。本机（Intel HD 4600 / D3D11）实测 60 fps。
3. `--remote-debugging-port` 要等 `/json/version` 返回 200 才能连，直接连会报
   `CDP 没就绪`。`shot_page.js` 里已经做了重试。
4. `curl` 访问 localhost 必须加 `--noproxy '*'`。
5. **视口必须显式覆盖**。headless Edge 有时不认 `--window-size`，`innerWidth` 只有 500，
   抽屉固定 392 px 就把主区挤到 108 px——量出来的布局全是错的。两个工具都已经用
   `Emulation.setDeviceMetricsOverride` 默认覆盖成 1440×900（`cdp_eval.js` 可用
   `CDP_W` / `CDP_H` 环境变量改）。

用 `tools/cdp_eval.js` 可以在页面里跑任意 JS 看运行时状态（`node tools/cdp_eval.js <url> "<expr>" [waitMs]`）。
注意 `sim.html` 里的 `ST` / `EMU` / `CAM` / `LINK` 是脚本顶层的 `const`，
在 `<script>`（非 module）里它们是全局的，CDP 能直接读到。

**如果只是人工看效果**，直接开浏览器访问 `http://127.0.0.1:8099/` 即可。

### 7.5 改电路后如何重新标定 REF

`body/mapping.py` 的 `REF` 字典是逐群手工标定的。改电路后必须重标：

```python
# 跑 8000 ms，每 20 ms 采一次各群 EMA 率，取中位和 p90
# 参考值取 (中位 + p90) / 2 附近，让约 50-60% 的时间输出在 0.3-1.0 之间
```

具体脚本可参考本文件 §5.2 的调试过程，或直接写个临时脚本跑 `net.rate[idx].mean()`。

---

## 8. 当前实测结果（作为基线）

### 脑行为（`tests/test_behaviour.py`，19 项全通过）

| 指标 | 结果 |
|---|---|
| 静息 | 0 尖峰（20 种子 × 400 ms） |
| loom（60 个神经元）→ 巨纤维 | **19/20 触发，延迟中位 4 ms** |
| 等规模非感觉刺激 → 巨纤维 | **4/20 触发**（判据 ≤8，比值 4.8x） |
| 实时倍率 | **75–82x**（单步 0.012–0.013 ms，numba 后端，随机器负载波动） |

### 身体（`tests/test_body.py`，26 项全通过）

| 指标 | 结果 |
|---|---|
| 输出帧数 | 450（50 Hz × 9 s） |
| 有动作的轴 | 6 个（L0 L1 L2 R0 R1 R2） |
| 逃逸触发 | 4 次 |
| TCode 指令 | 449 条 |

### 闭环 20 秒实测（`brain_loop.py --ms 20000`）

| 轴 | 范围 |
|---|---|
| L0 上下 | 0.500 → 0.965 |
| L1 前后 | **0.195 → 0.834**（前进后退都清晰） |
| L2 左右 | 0.177 → 0.860 |
| R0 旋转 | 0.274 → 0.752 |
| R1 滚转 | 0.233 → 0.773 |
| R2 俯仰 | 0.500 → 0.857 |
| V0 振动 | 0.000 → 0.927 |

刺激事件 13 次，逃逸触发 8 次 / 1000 帧，loom 同步峰值 60。端到端实时倍率 14.1x（耗时 1,417 ms / 20,000 ms）。

---

## 9. 六轴映射表（完整）

OSR6 六轴：`L0` 上下、`L1` 前后、`L2` 左右、`R0` 旋转、`R1` 滚转、`R2` 俯仰，取值 [0,1)，中位 0.5。

| 轴 | 公式（`body/mapping.py`） | 生物对应 |
|---|---|---|
| L0 | `0.5 + 0.20·fwd + 0.40·wing − 0.48·esc` | 振翅产生升力，逃逸时压低 |
| L1 | `0.5 − 0.42·back`（latch）或 `0.5 + 0.35·fwd + 0.45·esc` | 倒退 / 前进，逃逸时前冲 |
| L2 | `0.5 + 0.40·steer` | 左右转向 |
| R0 | `0.5 + 0.28·steer` | 转向时身体旋转 |
| R1 | `0.5 + 0.35·groom_wave` | 梳理时的滚转摆动 |
| R2 | `0.5 + 0.40·wing − 0.20·esc` | 振翅俯仰 |
| V0 | `(200·wing + 45·esc) / 250` | 翅拍频率代理（`wing` 满驱动 ≈ 200 Hz，上限 250 Hz） |

> `V0` 不是随手的混合量：`wing` 是翅运动群相对 `REF` 的归一化放电率，定义为
> 「翅拍驱动强度」。满驱动对应果蝇常态飞行的 ~200 Hz，生理上限 ~250 Hz，所以
> `V0 = 翅拍Hz / 250`（0.8 ≈ 200 Hz）。逃逸起飞时翅拍短暂加速（wing-assisted
> jump）再 +45 Hz。要看绝对频率，`V0 × 250` 就是 Hz 的近似值。

三条工程约束（来自方案调研）：
1. **转向只用变化量** —— 8 秒慢基线，只取瞬态（真实果蝇转向是脉冲式的）
2. **迟滞 + 驻留** —— `back_latch` 需要 ≥0.6 且驻留 80 ms
3. **强制钳位 + 输出平滑** —— 全部钳到 [0,1]，110 ms 平滑（真实舵机不能瞬变）

---

## 10. 许可注意事项（重要）

| 内容 | 许可 | 影响 |
|---|---|---|
| FlyWire FAFB v783 数据 | **CC BY-NC 4.0** | **禁止商业使用** |
| `data/circuit.json`（派生数据） | 继承 CC BY-NC 4.0 | 同上 |
| `body/web/vendor/osr-emu.dist.min.js` | 来自原 OSR6 项目（MIT） | 需保留原许可声明 |
| `body/osr/osr.py`、`osr_emu.py` | 来自原 OSR6 项目 | 同上 |
| 本项目原创代码 | 未定 | 建议加 LICENSE 文件 |

**接真机或商用前必须先定许可。** 原 OSR6 仓库目前没有 LICENSE 文件。

---

## 11. 常用命令速查

```powershell
# 一键启动（推荐，不受执行策略限制，自己找有 numpy 的 Python）
.\start.bat
.\start.bat 20000            # 跑 20 秒
.\start.bat 20000 8099 wind  # 时长 端口 场景

# 注意：本机 PATH 上的 `python` 没有 numpy，用 `py -3`
# 重新抽电路（partners 越大越慢越大）
py -3 etl.py --partners 30

# 纯脑演示
py -3 demo_brain.py --ms 900 --loom-at 300 --every 20

# 闭环，无界面
py -3 brain_loop.py --ms 20000

# 闭环 + 可视化
py -3 brain_loop.py --ms 120000 --web --port 8099

# 换场景（默认 live；demo 行为相同；quiet/flow/wind 无捕食者）
py -3 brain_loop.py --scenario live     # 正式默认：捕食者 + 手动刺激
py -3 brain_loop.py --scenario demo     # 演示（同样有捕食者）
py -3 brain_loop.py --scenario quiet    # 无刺激
py -3 brain_loop.py --scenario wind     # 只有风

# 关掉本体感觉回流
py -3 brain_loop.py --feedback 0

# 重新下载原始数据（约 55 MB）
py -3 tools\fetch.py --parts 4 classification.csv.gz coordinates.csv.gz consolidated_cell_types.csv.gz connections.csv.gz

# 测试（没装 pytest，直接跑）
py -3 tests\test_behaviour.py    # 19 项
py -3 tests\test_body.py         # 26 项

# 截图（先单独起一个带 CDP 的 Edge，见 §7.4）
node tools\shot_page.js "http://127.0.0.1:8099/" shot.png 7000 1440 900
```

---

## 12. 一句话给未来的自己

**脑的部分是真的，身体的部分是真的，中间那层映射是手工调出来的。**

2026-10-29 补完两条感觉通路（`forward` §4.7、`JO` §4.8）之后，六轴的每根轴都有对应的真实神经元群在推。
剩下三个缺口按价值排序：**① 转向/梳理/振翅没有感觉上游**（照 §4.8 的套路补）；
**② 真机实测**（入口已接好，见 §4.9 与 `body/device.py`，插上设备选端口即可，但一次都没真跑过）；
**③ 扩规模**——但这个要先改归一化（§6.2），否则会像 §6.1 那样把选择性搞没。

不要再花时间调 `mapping.py` 的系数了，系数调参的天花板已经到了。
也不要给弱通路加增益——§4.8 的增益扫描证明了那只会把噪声一起放大。
