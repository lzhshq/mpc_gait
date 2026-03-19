## 最终交付包（A1，speed=0.4）

本文件夹来自一次稳定运行的完整输出（12秒）：
- 速度目标：`--speed 0.4`
- 控制：`--controller joint_trot`

### 文件说明
- `log.csv`: 每个仿真步的数据（τ(t)、ω(t)、接触力真值/估计、接触状态估计等）
- `meta.json`: 运行参数与模型信息
- `plots/torque.png`: 关节力矩曲线
- `plots/joint_speed.png`: 关节角速度曲线
- `plots/contact_force_truth_vs_est.png`: 接触力真值 vs 估计
- `plots/contact_state.png`: 接触状态（估计）对比
- `plots/stats.json`: peak / RMS / mean 统计

### 复现命令（在项目根目录）
```bash
python -m src.sim_run_mujoco --model_preset a1 --controller joint_trot --gui 1 --duration 12 --speed 0.4 --log_dir runs/final_run_speed0p4
python -m src.analyze --log_dir runs/final_run_speed0p4 --out_dir runs/final_run_speed0p4/plots
```

