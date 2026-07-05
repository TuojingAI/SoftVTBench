import numpy as np
import matplotlib.pyplot as plt


class ForcePositionDebugVisualizer:
    """简单的力–位混合调试可视化工具。

    - 预测量（来自 HDF5 / action）：红色
    - 实测量（来自仿真传感器）：蓝色
    - 位姿预测（eef 位置）：紫色（目前没有 delta，预留蓝色通道）
    """

    def __init__(self):
        plt.ion()
        self.plt = plt
        self.fig, self.axs = plt.subplots(3, 3, figsize=(14, 9))
        self.t = []
        self._step = 0

        # 预先准备几组时间序列（只负责画图，统计在 replay 脚本中完成）
        self.series: dict[str, list] = {
            # 外力在 base 坐标系下的各分量
            "Fext_x_pred": [],
            "Fext_x_meas": [],
            "Fext_y_pred": [],
            "Fext_y_meas": [],
            "Fext_z_pred": [],
            "Fext_z_meas": [],
            # 挤压力与夹爪
            "fsq_pred": [],
            "fsq_meas": [],
            "d_pred": [],
            "d_actual": [],
            # EEF 位置（预测 vs 混合后）
            "eef_x_pred": [],
            "eef_x_actual": [],
            "eef_y_pred": [],
            "eef_y_actual": [],
            "eef_z_pred": [],
            "eef_z_actual": [],
        }

    def reset(self):
        """在新 episode 开始时重置时间轴和所有序列."""
        self.t = []
        self._step = 0
        for key in self.series:
            self.series[key].clear()

    def update(self, debug: dict):
        """根据 ForcePositionAction.debug_info 更新曲线."""
        self._step += 1
        self.t.append(self._step)
        t = np.array(self.t)

        # 从 debug dict 中取出需要画的量（env0 已经在 ActionTerm 中处理成 numpy）
        F_ext_pred_b = debug.get("F_ext_pred_b", np.zeros(3))
        F_ext_meas_b = debug.get("F_ext_meas_b", np.zeros(3))
        f_sq_pred = debug.get("f_sq_pred", 0.0)
        f_sq_meas = debug.get("f_sq_meas", 0.0)
        d_pred = debug.get("d_pred", 0.0)
        d_actual = debug.get("d_actual", 0.0)
        eef_pos_pred = debug.get("eef_pos_pred", np.zeros(3))
        eef_pos_delta = debug.get("eef_pos_delta", np.zeros(3))

        # 由 (pred, delta) 还原 hybrid 实际目标位置
        eef_pos_actual = eef_pos_pred + eef_pos_delta

        # 外力分量
        self.series["Fext_x_pred"].append(float(F_ext_pred_b[0]))
        self.series["Fext_x_meas"].append(float(F_ext_meas_b[0]))
        self.series["Fext_y_pred"].append(float(F_ext_pred_b[1]))
        self.series["Fext_y_meas"].append(float(F_ext_meas_b[1]))
        self.series["Fext_z_pred"].append(float(F_ext_pred_b[2]))
        self.series["Fext_z_meas"].append(float(F_ext_meas_b[2]))

        # 挤压力与夹爪
        fsq_pred_val = float(f_sq_pred)
        fsq_meas_val = float(f_sq_meas)
        self.series["fsq_pred"].append(fsq_pred_val)
        self.series["fsq_meas"].append(fsq_meas_val)
        self.series["d_pred"].append(float(d_pred))
        self.series["d_actual"].append(float(d_actual))
        self.series["eef_x_pred"].append(float(eef_pos_pred[0]))
        self.series["eef_x_actual"].append(float(eef_pos_actual[0]))
        self.series["eef_y_pred"].append(float(eef_pos_pred[1]))
        self.series["eef_y_actual"].append(float(eef_pos_actual[1]))
        self.series["eef_z_pred"].append(float(eef_pos_pred[2]))
        self.series["eef_z_actual"].append(float(eef_pos_actual[2]))

        # 绘制各子图
        # 1. 外力 x 分量（base frame）
        ax = self.axs[0, 0]
        ax.cla()
        ax.plot(t, self.series["Fext_x_pred"], "r-", label="Fx_pred")
        ax.plot(t, self.series["Fext_x_meas"], "b-", label="Fx_meas")
        ax.set_title("F_ext^b x (pred vs meas)")
        ax.legend(loc="upper right")

        # 2. 外力 y 分量（base frame）
        ax = self.axs[0, 1]
        ax.cla()
        ax.plot(t, self.series["Fext_y_pred"], "r-", label="Fy_pred")
        ax.plot(t, self.series["Fext_y_meas"], "b-", label="Fy_meas")
        ax.set_title("F_ext^b y (pred vs meas)")
        ax.legend(loc="upper right")

        # 3. 外力 z 分量（base frame）
        ax = self.axs[0, 2]
        ax.cla()
        ax.plot(t, self.series["Fext_z_pred"], "r-", label="Fz_pred")
        ax.plot(t, self.series["Fext_z_meas"], "b-", label="Fz_meas")
        ax.set_title("F_ext^b z (pred vs meas)")
        ax.legend(loc="upper right")

        # 4. 抓取挤压力标量
        ax = self.axs[1, 0]
        ax.cla()
        ax.plot(t, self.series["fsq_pred"], "r-", label="fsq_pred")
        ax.plot(t, self.series["fsq_meas"], "b-", label="fsq_meas")
        ax.set_title("Squeeze force scalar")
        ax.legend(loc="upper right")

        # 5. 夹爪开合度
        ax = self.axs[1, 1]
        ax.cla()
        ax.plot(t, self.series["d_pred"], "r-", label="d_pred")
        ax.plot(t, self.series["d_actual"], "b-", label="d_actual")
        ax.set_title("Gripper opening (pred vs actual)")
        ax.legend(loc="upper right")

        # 6. 预留其它分量/调试信息
        ax = self.axs[1, 2]
        ax.cla()
        ax.axis("off")
        ax.text(0.05, 0.8, "Force-Position Debug", fontsize=10, transform=ax.transAxes)

        # 7. EEF x 位置（预测 vs 实际混合后）
        ax = self.axs[2, 0]
        ax.cla()
        ax.plot(t, self.series["eef_x_pred"], color="purple", label="x_pred")
        ax.plot(t, self.series["eef_x_actual"], color="blue", label="x_actual")
        ax.set_title("EEF x (pred vs actual)")
        ax.legend(loc="upper right")

        # 8. EEF y 位置（预测 vs 实际混合后）
        ax = self.axs[2, 1]
        ax.cla()
        ax.plot(t, self.series["eef_y_pred"], color="purple", label="y_pred")
        ax.plot(t, self.series["eef_y_actual"], color="blue", label="y_actual")
        ax.set_title("EEF y (pred vs actual)")
        ax.legend(loc="upper right")

        # 9. EEF z 位置（预测 vs 实际混合后）
        ax = self.axs[2, 2]
        ax.cla()
        ax.plot(t, self.series["eef_z_pred"], color="purple", label="z_pred")
        ax.plot(t, self.series["eef_z_actual"], color="blue", label="z_actual")
        ax.set_title("EEF z (pred vs actual)")
        ax.legend(loc="upper right")

        self.fig.tight_layout()
        self.plt.pause(0.001)

    def close(self):
        """关闭可视化窗口."""
        self.plt.ioff()
        self.plt.close(self.fig)


