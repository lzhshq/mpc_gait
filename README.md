## 视频演示
<video src="https://private-user-images.githubusercontent.com/220468817/566067194-bedf8e49-6356-4b29-aa13-bff33885c9d8.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzM4OTk2OTIsIm5iZiI6MTc3Mzg5OTM5MiwicGF0aCI6Ii8yMjA0Njg4MTcvNTY2MDY3MTk0LWJlZGY4ZTQ5LTYzNTYtNGIyOS1hYTEzLWJmZjMzODg1YzlkOC5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMzE5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDMxOVQwNTQ5NTJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05NTA2MjY1YTU4YWQ5ODM3Y2NmMzUwMjRlMzJkNGQ5YzRjMjllMTdiNDA1ZWViOGMyNTllN2M1YzJjMzQ0YTQ3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.B-XtYHrgV2fm6_a6hfPRpDnGhRzjiHLv0PBB3xRcBAQ" controls width="700" autoplay loop muted playsinline>
  你的浏览器不支持HTML5视频播放，请点击<a href="https://private-user-images.githubusercontent.com/220468817/566067194-bedf8e49-6356-4b29-aa13-bff33885c9d8.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzM4OTk2OTIsIm5iZiI6MTc3Mzg5OTM5MiwicGF0aCI6Ii8yMjA0Njg4MTcvNTY2MDY3MTk0LWJlZGY4ZTQ5LTYzNTYtNGIyOS1hYTEzLWJmZjMzODg1YzlkOC5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMzE5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDMxOVQwNTQ5NTJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05NTA2MjY1YTU4YWQ5ODM3Y2NmMzUwMjRlMzJkNGQ5YzRjMjllMTdiNDA1ZWViOGMyNTllN2M1YzJjMzQ0YTQ3JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.B-XtYHrgV2fm6_a6hfPRpDnGhRzjiHLv0PBB3xRcBAQ">这里</a>下载视频
</video>
最终提交文件夹在：
- `deliverables/final_a1_speed0p4/`

## 1. 环境要求

- Windows 10/11
- Python 3.10+（推荐 3.11）

安装依赖（在本目录下执行）：

```bash
pip install -r requirements.txt
```

---


### 2.A1

```bash
# 速度 0.4（最终提交版本）
python -m src.sim_run_mujoco --model_preset a1 --controller joint_trot --gui 1 --duration 12 --speed 0.4 --log_dir runs/a1_demo
python -m src.analyze --log_dir runs/a1_demo --out_dir runs/a1_demo/plots
```



## 3. 自动画图 + 统计（峰值 / RMS / 平均）

```bash
python -m src.analyze --log_dir runs/run1 --out_dir runs/run1/plots
```

输出：
- `plots/torque.png`、`plots/joint_speed.png`
- `plots/contact_force_truth_vs_est.png`
- `plots/contact_state.png`
- `stats.json`：每个关节/每条腿的 peak / rms / mean

---

## 4. 交付物对应关系

- **MPC / 规划器方案说明**：见 `src/controller_qp.py`（QP 力分配 + 足端轨迹 + 力矩生成）
- **稳定步态仿真结果（截图/视频帧）**：运行 `--gui 1` 录屏即可
- **力矩/转速/接触力曲线**：`src/analyze.py` 自动生成
- **接触状态与接触力估计 + 误差分析**：`src/contact_estimation.py` + `src/analyze.py`
- **原始数据与自动化脚本**：`runs/*` + `src/analyze.py`

---

## 5. 常用参数

`src/sim_run.py` 主要参数：
- `--duration`：仿真时长（秒）
- `--speed`：期望直线速度（m/s）
- `--gui`：1 开启界面；0 关闭
- `--log_dir`：输出目录
- `--terrain`：`plane` 或 `rough`（可选）

---

## 6. “任务2必做”：接触状态/接触力估计用的是什么方法？

### 6.1 接触真值（对照）
MuJoCo 可以直接从接触点读取接触力（作为对照真值）。本项目在 `src/sim_run_mujoco.py` 中对每个足端 geom 汇总接触力，并写入 `log.csv` 的 `f_truth_*` 字段。

### 6.2 接触状态估计（stance/swing）
在本项目里用的是**力矩+加速度的阈值判别**（工程上常用的启发式方法）：
- 先用有限差分得到关节角加速度 \(\ddot{q}\)
- 对每条腿，根据当前腿的雅可比 \(J\) 与关节力矩 \(\tau\)，估计足端力 \(f_{est}\)
- 若 \(|f_{est,z}|\)（或 \(\|f_{est}\|\)）超过阈值，则判定 stance，否则 swing


### 6.3 接触力估计（Jacobian 反算）
支撑腿在准静态/低速下近似：
\[
\tau \approx J^T f
\]
所以：
\[
f_{est} \approx (J^T)^{+} \tau
\]
然后将 \(f_{est}\) 与仿真读取的 \(f_{truth}\) 做对比，输出误差曲线和统计。

### 6.4 代码位置
- **真值接触力读取**：`src/sim_run_mujoco.py` 中 `_foot_truth_contact_force_world(...)` + 主循环里写入 `f_truth_*`
- **接触力估计（Jacobian 反算）**：`src/contact_estimation.py` 的 `estimate_contact_forces_from_tau(...)`（输出写入 `f_est_*`）
- **接触状态估计（stance/swing）**：`src/contact_estimation.py` 的 `estimate_contact_state_from_force(...)`（输出写入 `contact_est_*`）
- **对比图/误差统计**：`src/analyze.py`（会生成 `contact_force_truth_vs_est.png`、`contact_state.png`、`stats.json`）

---


## 7. 目录结构

```
task2_mpc_gait/
  requirements.txt
  README.md
  deliverables
  src/
    sim_run.py
    controller_qp.py
    contact_estimation.py
    analyze.py
    utils_math.py
```

