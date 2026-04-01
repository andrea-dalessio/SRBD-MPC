import numpy as np
from matplotlib import pyplot as plt

class Logger():
    def __init__(self, initial):
        self.log = {}
        for item in initial.keys():
            for level in initial[item].keys():
                self.log['desired', item, level] = []
                self.log['current', item, level] = []
        
        self.initial_plan = None
        self.post_impact_plan = None

    def log_footsteps(self, initial_plan, post_impact_plan):
        self.initial_plan = initial_plan
        self.post_impact_plan = post_impact_plan


    def log_data(self, desired, current, forces=None, commands=None):
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

    def show_all_plots(self):
        print("Visualizzazione dei grafici in corso...")
        
        # Estrazione Dati
        com_des = np.array(self.log['desired', 'com', 'pos'])
        com_cur = np.array(self.log['current', 'com', 'pos'])
        zmp_des = np.array(self.log['desired', 'zmp', 'pos'])
        zmp_cur = np.array(self.log['current', 'zmp', 'pos'])
        quat_des = np.array(self.log['desired', 'base', 'quat'])
        quat_cur = np.array(self.log['current', 'base', 'quat'])
        
        time_steps = np.arange(len(com_des))
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Analisi SRBD-MPC ed Inverse Dynamics', fontsize=16)

        # 1. CoM Tracking (X, Y, Z)
        ax = axs[0, 0]
        ax.plot(time_steps, com_des[:, 0], 'r--', label='X Des')
        ax.plot(time_steps, com_cur[:, 0], 'r-', label='X Cur')
        ax.plot(time_steps, com_des[:, 1], 'g--', label='Y Des')
        ax.plot(time_steps, com_cur[:, 1], 'g-', label='Y Cur')
        ax.plot(time_steps, com_des[:, 2], 'b--', label='Z Des')
        ax.plot(time_steps, com_cur[:, 2], 'b-', label='Z Cur')
        ax.set_title('CoM Tracking (Metri)')
        ax.legend(loc='lower left', prop={'size': 7})
        ax.grid(True)

        # 2. Orientamento Base (Quaternioni XYZ)
        ax = axs[0, 1]
        # In SRBD il controllo d'assetto sostituisce lo ZMP del LIP.
        # Plottiamo le componenti X, Y, Z del quaternione del torso per vedere la stabilità.
        ax.plot(time_steps, quat_cur[:, 1], label='Qx (Roll base)')
        ax.plot(time_steps, quat_cur[:, 2], label='Qy (Pitch base)')
        ax.plot(time_steps, quat_cur[:, 3], label='Qz (Yaw base)')
        ax.set_title('Base Orientation (Quaternions Tracking)')
        ax.set_ylabel('Quat Value')
        ax.legend(loc='lower left')
        ax.grid(True)

        # 3. Forze Verticali (GRF)
        ax = axs[1, 0]
        if 'forces' in self.log and len(self.log['forces']) > 0:
            forces = np.array(self.log['forces']) # Shape (T, 24)
            # Ricordiamo che U è [fx,fy,fz] x 4 per Left e x 4 per Right
            # Fz totale = somma degli indici 2, 5, 8, 11 (Left) e 14, 17, 20, 23 (Right)
            fz_left = np.sum(forces[:, [2, 5, 8, 11]], axis=1)
            fz_right = np.sum(forces[:, [14, 17, 20, 23]], axis=1)
            fz_tot = fz_left + fz_right
            ax.plot(time_steps, fz_tot, 'k-', label='Fz Totale (~376 N)')
            ax.plot(time_steps, fz_left, 'b-', label='Fz Left', alpha=0.7)
            ax.plot(time_steps, fz_right, 'r-', label='Fz Right', alpha=0.7)
            ax.set_title('Ground Reaction Forces Z (Newton)')
            ax.legend()
        else:
            ax.set_title('Forze non loggate')
        ax.grid(True)

        # 4. Coppie WBC Inverse Dynamics
        ax = axs[1, 1]
        if 'commands' in self.log and len(self.log['commands']) > 0:
            commands = np.array(self.log['commands'])
            # Mostriamo le coppie delle prime 6 giunture loggate (es. anca e caviglia)
            for i in range(min(6, commands.shape[1])):
                ax.plot(time_steps, commands[:, i], label=f'Joint {i} Tau')
            ax.set_title('Inverse Dynamics Torques (WBC)')
            ax.set_ylabel('Nm')
            ax.legend(prop={'size': 7})
        else:
            ax.set_title('Torques non loggati')
        ax.grid(True)

        plt.tight_layout()
        
        # New figure for footsteps
        if hasattr(self, 'initial_plan') and self.initial_plan is not None:
            fig2, ax2 = plt.subplots(figsize=(8, 10))
            fig2.suptitle('Footstep Replanning (2D Map)', fontsize=16)

            # Plotta i footprint iniziali
            for i, step in enumerate(self.initial_plan):
                x, y, z = step['pos']
                color = 'blue' if step['foot_id'] == 'lfoot' else 'blue'
                ax2.plot(x, y, marker='s', markersize=30, color=color, alpha=0.2)
                ax2.text(x, y, str(i), color=color, fontsize=14, ha='center', va='center', fontweight='bold')

            # Plotta i footprint post impatto
            if hasattr(self, 'post_impact_plan') and self.post_impact_plan is not None:
                for i, step in enumerate(self.post_impact_plan):
                    x, y, z = step['pos']
                    ax2.plot(x, y, marker='s', markersize=30, markeredgecolor='red', markerfacecolor='none', linestyle='--', linewidth=3)
                    ax2.text(x, y+0.04, f"{i}'", color='red', fontsize=14, ha='center', va='center', fontweight='bold')

            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Confronto Traiettoria (Trasparente: nominale | Bordata Rossa: ri-pianificata)')
            ax2.grid(True)
            ax2.axis('equal')
            fig2.tight_layout()

        # New figure for Torso vs Feet Yaw
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        fig3.suptitle('Torso Yaw vs Feet Yaw Tracking', fontsize=14)
        
        w, x, y, z = quat_cur[:, 0], quat_cur[:, 1], quat_cur[:, 2], quat_cur[:, 3]
        base_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
        
        lfoot_cur = np.array(self.log['current', 'lfoot', 'pos'])
        rfoot_cur = np.array(self.log['current', 'rfoot', 'pos'])
        lfoot_yaw = lfoot_cur[:, 2]
        rfoot_yaw = rfoot_cur[:, 2]
        avg_feet_yaw = (lfoot_yaw + rfoot_yaw) / 2.0
        
        ax3.plot(time_steps, base_yaw, 'b-', linewidth=2, label='Torso Yaw (Current)')
        ax3.plot(time_steps, avg_feet_yaw, 'k--', linewidth=2, label='Average Feet Yaw (Target)')
        ax3.plot(time_steps, lfoot_yaw, 'g-', alpha=0.5, label='Left Foot Yaw')
        ax3.plot(time_steps, rfoot_yaw, 'r-', alpha=0.5, label='Right Foot Yaw')
        
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Yaw (Radians)')
        ax3.set_title('Confronto Orientamento Torso e Piedi')
        ax3.grid(True)
        ax3.legend()
        fig3.tight_layout()

        plt.show()