import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import os

class Logger():
    def __init__(self, initial):
        self.log = {}
        for item in initial.keys():
            for level in initial[item].keys():
                self.log['desired', item, level] = []
                self.log['current', item, level] = []

        self.log['disturbances'] = []
        
        self.initial_plan = None
        self.post_impact_plan = None

    def log_footsteps(self, initial_plan, post_impact_plan, actual_footsteps=None):
        self.initial_plan = initial_plan
        self.actual_plan = post_impact_plan          # planned/replanned steps for the map
        self.actual_footsteps = actual_footsteps if actual_footsteps is not None else []

    def log_disturbance(self, time_stamp, force_world, forward_xy, com_xy):
        self.log['disturbances'].append({
            'time': float(time_stamp),
            'force': np.array(force_world, dtype=float),
            'forward_xy': np.array(forward_xy, dtype=float),
            'com_xy': np.array(com_xy, dtype=float)
        })


    def log_data(self, desired, current, forces=None, commands=None, measured_forces=None):
        for item in desired.keys():
            for level in desired[item].keys():
                self.log['desired', item, level].append(desired[item][level])
                self.log['current', item, level].append(current[item][level])
        
        if forces is not None:
            if 'forces' not in self.log:
                self.log['forces'] = []
            self.log['forces'].append(forces)
            
        if commands is not None:
            if 'commands' not in self.log:
                self.log['commands'] = []
            self.log['commands'].append(commands)

        if measured_forces is not None:
            if 'measured_forces' not in self.log:
                self.log['measured_forces'] = []
            self.log['measured_forces'].append(measured_forces)

    @staticmethod
    def _quat_wxyz_to_rpy_deg(quat_array):
        quat = np.asarray(quat_array, dtype=float)
        if quat.ndim != 2 or quat.shape[1] != 4:
            raise ValueError("Quaternion array must have shape (N, 4)")

        norm = np.linalg.norm(quat, axis=1, keepdims=True)
        norm = np.where(norm < 1e-12, 1.0, norm)
        quat = quat / norm

        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch_arg = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(pitch_arg, -1.0, 1.0))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yaw = np.unwrap(yaw)

        return np.rad2deg(np.vstack((roll, pitch, yaw)).T)

    def _extract_disturbance_impulses(self):
        events = self.log.get('disturbances', [])
        if len(events) <= 1:
            return events

        times = np.array([e['time'] for e in events], dtype=float)
        dt = np.diff(times)
        dt = dt[dt > 1e-7]
        nominal_dt = np.median(dt) if dt.size > 0 else 0.01
        gap_threshold = max(2.5 * nominal_dt, 1e-3)

        impulses = [events[0]]
        for i in range(1, len(events)):
            if (times[i] - times[i - 1]) > gap_threshold:
                impulses.append(events[i])

        return impulses

    @staticmethod
    def _moving_average(signal, window):
        arr = np.asarray(signal, dtype=float)
        if arr.size == 0:
            return arr

        win = max(1, int(window))
        if win == 1 or arr.size < 3:
            return arr.copy()

        win = min(win, arr.size)

        kernel = np.ones(win, dtype=float) / float(win)
        pad_left = win // 2
        pad_right = win - 1 - pad_left
        padded = np.pad(arr, (pad_left, pad_right), mode='edge')
        return np.convolve(padded, kernel, mode='valid')

    def show_all_plots(self, save_dir=None, show_plots=True):
        print("Visualizzazione dei grafici in corso...")
        saved_files = []

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        def save_figure(fig, filename):
            if save_dir is None:
                return
            out_path = os.path.join(save_dir, filename)
            fig.savefig(out_path, dpi=180, bbox_inches='tight')
            saved_files.append(out_path)
        
        # Data extraction
        com_des = np.array(self.log['desired', 'com', 'pos'])
        if len(com_des) == 0:
            print("Nessun campione disponibile nel logger.")
            return saved_files

        com_cur = np.array(self.log['current', 'com', 'pos'])
        quat_des = np.array(self.log['desired', 'base', 'quat'])
        quat_cur = np.array(self.log['current', 'base', 'quat'])
        zmp_cur = None
        zmp_des = None
        if ('current', 'zmp', 'pos') in self.log and len(self.log['current', 'zmp', 'pos']) > 0:
            zmp_cur = np.array(self.log['current', 'zmp', 'pos'])
        if ('desired', 'zmp', 'pos') in self.log and len(self.log['desired', 'zmp', 'pos']) > 0:
            zmp_des = np.array(self.log['desired', 'zmp', 'pos'])
        time_steps = np.arange(len(com_des))

        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'legend.fontsize': 9
        })
        contact_ma_window = int(os.environ.get('PLOT_CONTACT_FORCE_MA_WINDOW', '20'))

        # Figure 1: Footsteps + CoM map + one impulse arrow per disturbance window
        if hasattr(self, 'initial_plan') and self.initial_plan is not None:
            fig1, ax1 = plt.subplots(figsize=(9.5, 10.0))
            fig1.suptitle('Footstep Map and Disturbance Impulses', fontsize=16)

            initial_labels_done = {'lfoot': False, 'rfoot': False}

            # Footprint physical dimensions (approximate)
            foot_l = 0.10
            foot_w = 0.10

            for i, step in enumerate(self.initial_plan):
                x, y, z = step['pos']
                
                # We need the yaw angle if available to rotate the rectangle properly.
                # However, for simplicity and since walking is mostly forward here, we just use axis-aligned boxes 
                # or extract yaw from step['ang'] if present.
                yaw = step.get('ang', [0, 0, 0])[2]
                
                color = 'tab:blue' if step['foot_id'] == 'lfoot' else 'tab:green'
                foot_lbl = 'Left Footsteps (nominal)' if step['foot_id'] == 'lfoot' else 'Right Footsteps (nominal)'
                
                # Calculate bottom-left corner from center
                bl_x = x - (foot_l / 2) * np.cos(yaw) + (foot_w / 2) * np.sin(yaw)
                bl_y = y - (foot_l / 2) * np.sin(yaw) - (foot_w / 2) * np.cos(yaw)

                rect = patches.Rectangle(
                    (bl_x, bl_y), foot_l, foot_w,
                    angle=np.degrees(yaw),
                    facecolor=color,
                    alpha=0.22,
                    edgecolor='none',
                    label=foot_lbl if not initial_labels_done[step['foot_id']] else None
                )
                ax1.add_patch(rect)
                initial_labels_done[step['foot_id']] = True
                ax1.text(x, y, str(i), color=color, fontsize=12, ha='center', va='center', fontweight='bold')

            if hasattr(self, 'actual_plan') and self.actual_plan is not None:
                actual_label_done = False
                for step in self.actual_plan:
                    i = step.get('step_idx', '')
                    x, y, z = step['pos']
                    yaw = step.get('ang', [0, 0, 0])[2]
                    bl_x = x - (foot_l / 2) * np.cos(yaw) + (foot_w / 2) * np.sin(yaw)
                    bl_y = y - (foot_l / 2) * np.sin(yaw) - (foot_w / 2) * np.cos(yaw)
                    rect = patches.Rectangle(
                        (bl_x, bl_y), foot_l, foot_w,
                        angle=np.degrees(yaw),
                        fill=False,
                        edgecolor='darkorange',
                        linewidth=2.2,
                        label='Footsteps (planned/replanned)' if not actual_label_done else None
                    )
                    ax1.add_patch(rect)
                    actual_label_done = True
                    ax1.text(x, y + 0.035, f"{i}p", color='darkorange', fontsize=10,
                             ha='center', va='center', fontweight='bold')

            # Actual executed footsteps from DART skeleton (global world-frame coordinates)
            # These come from getTransform() queries on l_sole / r_sole — ground-truth kinematics.
            if hasattr(self, 'actual_footsteps') and self.actual_footsteps:
                dart_label_done = False
                for contact in self.actual_footsteps:
                    landed = contact.get('landed_foot')
                    step_lbl = str(contact.get('step_idx', '?'))
                    feet_to_show = [landed] if landed in ('lfoot', 'rfoot') else ['lfoot', 'rfoot']
                    for foot in feet_to_show:
                        pos = contact.get(foot)
                        if pos is None:
                            continue
                        fx, fy = float(pos[0]), float(pos[1])
                        bl_x = fx - foot_l / 2
                        bl_y = fy - foot_w / 2
                        rect = patches.Rectangle(
                            (bl_x, bl_y), foot_l, foot_w,
                            angle=0.0,
                            fill=False,
                            edgecolor='red',
                            linewidth=2.2,
                            label='Footsteps (actual — DART skeleton)' if not dart_label_done else None
                        )
                        ax1.add_patch(rect)
                        dart_label_done = True
                        ax1.text(fx, fy + 0.035, step_lbl, color='red', fontsize=11,
                                 ha='center', va='center', fontweight='bold')

            ax1.plot(com_cur[:, 0], com_cur[:, 1], 'k-', linewidth=2.3, label='CoM Current')
            ax1.plot(com_des[:, 0], com_des[:, 1], 'k--', linewidth=1.7, alpha=0.9, label='CoM Desired')
            if zmp_cur is not None:
                # Robustify CoP trace: remove extreme outliers by clipping to percentiles
                try:
                    x = zmp_cur[:, 0].astype(float)
                    y = zmp_cur[:, 1].astype(float)
                    # percentile clipping to remove impulses
                    lo_x, hi_x = np.percentile(x, [1.0, 99.0])
                    lo_y, hi_y = np.percentile(y, [1.0, 99.0])
                    x_clipped = np.clip(x, lo_x, hi_x)
                    y_clipped = np.clip(y, lo_y, hi_y)
                    # Smooth with moving average (same window used for contact forces)
                    x_s = self._moving_average(x_clipped, contact_ma_window)
                    y_s = self._moving_average(y_clipped, contact_ma_window)
                    # Adjust lengths if moving_average pads
                    n = min(len(x_s), len(y_s))
                    ax1.plot(x_s[:n], y_s[:n], color='tab:red', linewidth=1.8, alpha=0.95, label='CoP Current (smoothed)')
                    ax1.scatter(x_s[0], y_s[0], color='tab:red', s=24, marker='o', label='CoP start')
                except Exception:
                    ax1.plot(zmp_cur[:, 0], zmp_cur[:, 1], color='tab:red', linewidth=1.8, alpha=0.95, label='CoP Current')
                    ax1.scatter(zmp_cur[0, 0], zmp_cur[0, 1], color='tab:red', s=24, marker='o', label='CoP start')
            if zmp_des is not None:
                try:
                    xd = zmp_des[:, 0].astype(float)
                    yd = zmp_des[:, 1].astype(float)
                    xd_cl = np.clip(xd, np.percentile(xd, 1.0), np.percentile(xd, 99.0))
                    yd_cl = np.clip(yd, np.percentile(yd, 1.0), np.percentile(yd, 99.0))
                    xd_s = self._moving_average(xd_cl, contact_ma_window)
                    yd_s = self._moving_average(yd_cl, contact_ma_window)
                    m = min(len(xd_s), len(yd_s))
                    ax1.plot(xd_s[:m], yd_s[:m], color='tab:red', linestyle='--', linewidth=1.2, alpha=0.65, label='CoP Desired (smoothed)')
                except Exception:
                    ax1.plot(zmp_des[:, 0], zmp_des[:, 1], color='tab:red', linestyle='--', linewidth=1.2, alpha=0.65, label='CoP Desired')

            impulse_events = self._extract_disturbance_impulses()
            if len(impulse_events) > 0:
                map_span = max(np.ptp(com_cur[:, 0]), np.ptp(com_cur[:, 1]), 0.4)
                arrow_len = max(0.08, 0.12 * map_span)
                for i, event in enumerate(impulse_events):
                    p = event['com_xy']
                    f = event['force'][:2]
                    fn = np.linalg.norm(f)
                    if fn > 1e-8:
                        f_dir = f / fn
                        ax1.quiver(
                            p[0],
                            p[1],
                            arrow_len * f_dir[0],
                            arrow_len * f_dir[1],
                            angles='xy',
                            scale_units='xy',
                            scale=1.0,
                            color='tab:orange',
                            width=0.006,
                            headwidth=3.8,
                            headlength=5.0,
                            label='Disturbance impulse (start of interval)' if i == 0 else None
                        )
                        ax1.text(
                            p[0],
                            p[1] + 0.03,
                            f"t={event['time']:.2f}s",
                            color='tab:orange',
                            fontsize=9,
                            ha='center'
                        )

            ax1.set_xlabel('X [m]')
            ax1.set_ylabel('Y [m]')
            ax1.set_title('Footstep Replanning Map')
            ax1.grid(True, alpha=0.35)
            ax1.axis('equal')
            ax1.legend(loc='best')
            fig1.tight_layout()
            save_figure(fig1, '1_footsteps_map.png')

        forces_per_contact = None
        force_source = None
        if 'measured_forces' in self.log and len(self.log['measured_forces']) > 0:
            measured = np.asarray(self.log['measured_forces'], dtype=float)
            if measured.ndim == 1:
                measured = measured.reshape(1, -1)
            if measured.shape[1] == 24:
                forces_per_contact = measured.reshape(-1, 8, 3)
                force_source = 'measured'

        if forces_per_contact is None and 'forces' in self.log and len(self.log['forces']) > 0:
            forces = np.asarray(self.log['forces'], dtype=float)
            if forces.ndim == 1:
                forces = forces.reshape(1, -1)
            if forces.shape[1] == 24:
                forces_per_contact = forces.reshape(-1, 8, 3)
                force_source = 'mpc'

        left_labels = ['L-P1 (+x,+y)', 'L-P2 (+x,-y)', 'L-P3 (-x,+y)', 'L-P4 (-x,-y)']
        right_labels = ['R-P1 (+x,+y)', 'R-P2 (+x,-y)', 'R-P3 (-x,+y)', 'R-P4 (-x,-y)']
        left_colors = ['#0B3C5D', '#1F77B4', '#2E86DE', '#4FC3F7']
        right_colors = ['#7F1D1D', '#C62828', '#EF5350', '#FF8A65']

        def plot_foot_reaction(fig_title, fig_name, contact_indices, labels, colors):
            fig, ax = plt.subplots(figsize=(12.8, 6.8))
            title_suffix = ' (measured from physics)' if force_source == 'measured' else ' (MPC command)'
            fig.suptitle(fig_title + title_suffix, fontsize=16)

            if forces_per_contact is None:
                if 'forces' not in self.log or len(self.log['forces']) == 0:
                    ax.set_title('Forze di reazione non loggate')
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"Formato forze non valido: atteso 24, trovato {np.asarray(self.log['forces']).shape[-1]}",
                        transform=ax.transAxes,
                        ha='center',
                        va='center',
                        fontsize=12
                    )
            else:
                tf = np.arange(forces_per_contact.shape[0])
                for local_i, contact_i in enumerate(contact_indices):
                    fz_smoothed = self._moving_average(forces_per_contact[:, contact_i, 2], contact_ma_window)
                    ax.plot(
                        tf,
                        fz_smoothed,
                        color=colors[local_i],
                        linewidth=2.1,
                        label=f"{labels[local_i]} - Fz MA({contact_ma_window})"
                    )

                foot_fz = forces_per_contact[:, contact_indices, 2]
                foot_fz_sum = np.sum(foot_fz, axis=1)
                foot_fz_sum_smoothed = self._moving_average(foot_fz_sum, contact_ma_window)
                ax.plot(
                    tf,
                    foot_fz_sum_smoothed,
                    color='#111111',
                    linewidth=2.8,
                    linestyle='--',
                    label=f"Sum(4 contacts) - Fz MA({contact_ma_window})"
                )
                ax.axhline(
                    400.0,
                    color='#666666',
                    linewidth=1.4,
                    linestyle=':',
                    label='Reference: 400 N'
                )

                foot_forces = forces_per_contact[:, contact_indices, 2]
                y_min = min(-15.0, float(np.min(foot_forces) * 1.05))
                y_max = max(80.0, float(np.max(foot_forces) * 1.08), float(np.max(foot_fz_sum) * 1.08), 520.0)
                ax.set_ylim(y_min, y_max)
                ax.legend(loc='upper right', ncol=1)

            ax.set_xlabel('Sample index')
            ax.set_ylabel('Vertical force Fz [N]')
            ax.grid(True, alpha=0.35)
            fig.tight_layout()
            save_figure(fig, fig_name)

        # Figure 2: Left foot contact reaction forces
        plot_foot_reaction(
            'Reaction Forces per Contact Point - Left Foot',
            '2_reaction_forces_left_foot.png',
            [0, 1, 2, 3],
            left_labels,
            left_colors
        )

        # Figure 3: Right foot contact reaction forces
        plot_foot_reaction(
            'Reaction Forces per Contact Point - Right Foot',
            '3_reaction_forces_right_foot.png',
            [4, 5, 6, 7],
            right_labels,
            right_colors
        )

        if forces_per_contact is not None:
            total_fz = np.sum(forces_per_contact[:, :, 2], axis=1)
            source_label = 'measured' if force_source == 'measured' else 'mpc'
            print(
                f"[LOGGER] Total Fz stats ({source_label}) [N] -> "
                f"min={np.min(total_fz):.2f}, mean={np.mean(total_fz):.2f}, max={np.max(total_fz):.2f}"
            )
            left_max = np.max(forces_per_contact[:, 0:4, 2], axis=0)
            right_max = np.max(forces_per_contact[:, 4:8, 2], axis=0)
            print("[LOGGER] Max Left-foot contacts Fz [N]: " + ", ".join([f"{v:.2f}" for v in left_max]))
            print("[LOGGER] Max Right-foot contacts Fz [N]: " + ", ".join([f"{v:.2f}" for v in right_max]))

            if force_source == 'mpc' and max(np.max(left_max), np.max(right_max)) > 580.0:
                print("[LOGGER][WARN] Picchi Fz oltre 580N rilevati: controllare limiti/saturazioni MPC.")

        # Figure 3: Body orientation (RPY) and CoM tracking
        fig3, (ax3_top, ax3_bottom) = plt.subplots(2, 1, figsize=(13, 9.0), sharex=True)
        fig3.suptitle('Body Orientation (RPY) and CoM Tracking', fontsize=16)

        rpy_des_deg = self._quat_wxyz_to_rpy_deg(quat_des)
        rpy_cur_deg = self._quat_wxyz_to_rpy_deg(quat_cur)

        rpy_names = ['Roll', 'Pitch', 'Yaw']
        rpy_colors = ['tab:red', 'tab:green', 'tab:blue']
        for idx, (name, color) in enumerate(zip(rpy_names, rpy_colors)):
            ax3_top.plot(time_steps, rpy_des_deg[:, idx], linestyle='--', color=color, linewidth=2.0, label=f'{name} desired')
            ax3_top.plot(time_steps, rpy_cur_deg[:, idx], linestyle='-', color=color, linewidth=1.8, alpha=0.9, label=f'{name} current')

        ax3_top.set_ylabel('Angle [deg]')
        ax3_top.set_title('Body Orientation Tracking (Roll/Pitch/Yaw)')
        ax3_top.grid(True, alpha=0.35)
        ax3_top.legend(loc='best', ncol=2)

        com_names = ['X', 'Y', 'Z']
        com_colors = ['tab:red', 'tab:green', 'tab:blue']
        for idx, (name, color) in enumerate(zip(com_names, com_colors)):
            ax3_bottom.plot(time_steps, com_des[:, idx], linestyle='--', color=color, linewidth=2.0, label=f'CoM {name} desired')
            ax3_bottom.plot(time_steps, com_cur[:, idx], linestyle='-', color=color, linewidth=1.8, alpha=0.9, label=f'CoM {name} current')

        ax3_bottom.set_xlabel('Sample index')
        ax3_bottom.set_ylabel('Position [m]')
        ax3_bottom.set_title('Center of Mass Tracking')
        ax3_bottom.grid(True, alpha=0.35)
        ax3_bottom.legend(loc='best', ncol=2)

        fig3.tight_layout()
        save_figure(fig3, '4_orientation_com_tracking_rpy.png')

        if len(saved_files) > 0:
            print("Plot salvati come immagini:")
            for path in saved_files:
                print(f" - {path}")

        if show_plots:
            plt.show()
        else:
            plt.close('all')

        return saved_files
