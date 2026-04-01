import numpy as np
import casadi as cs

class SrbdMpc:
    def __init__ (self, initial, footstep_planner, params):
        # parameters
        self.params = params
        self.N = params['N']
        self.delta = params['world_time_step']
        self.foot_size = params['foot_size']
        self.initial = initial
        self.quat_ref = np.array(initial['base']['quat']).copy()
        self.footstep_planner = footstep_planner
        self.mass = params['mass']
        self.I = params['inertia'] 
        self.I_body_inv = np.linalg.inv(self.I)
        self.mu = params['µ']
        self.g = [0, 0, -params['g']]
        
        # Definizione della dinamica f con rotazione dell'inerzia
        self.f = lambda x, u, p_contacts: self._get_dynamics_with_rot_inertia(x, u, p_contacts)
        self.last_X = None
        self.last_U = None
        self.last_solve_success = True
        self.last_solve_message = ""

    def _is_finite_state(self, state):
        state_arrays = [
            state['com']['pos'],
            state['com']['vel'],
            state['base']['quat'],
            state['base']['omega']
        ]
        return all(np.all(np.isfinite(np.asarray(arr))) for arr in state_arrays)

    def _build_safe_fallback(self, current_state, next_step_target):
        optimal_controls = np.zeros(24)
        fz_each = (self.mass * abs(self.g[2])) / 8.0
        for i in range(8):
            optimal_controls[i*3 + 2] = fz_each

        target_state = {
            'com': {
                'pos': current_state['com']['pos'].copy(),
                'vel': current_state['com']['vel'].copy(),
                'acc': np.zeros(3)
            },
            'base': {
                'quat': current_state['base']['quat'].copy(),
                'omega': current_state['base']['omega'].copy()
            }
        }

        self.last_X = np.zeros((13, self.N + 1))
        for k in range(self.N + 1):
            self.last_X[0:3, k] = target_state['com']['pos']
            self.last_X[6:10, k] = target_state['base']['quat']

        self.last_U = np.zeros((24, self.N))
        for k in range(self.N):
            for i in range(8):
                self.last_U[i*3 + 2, k] = fz_each

        self.last_p_swing = np.array(next_step_target).copy()
        return optimal_controls, target_state

    def _get_dynamics_with_rot_inertia(self, x, u, p_contacts):
        q = x[6:10]      
        omega = x[10:13] 
        
        R = self.quat_to_rot(q)
        

        I_world_inv = R @ self.I_body_inv @ R.T
        
        # Ci serve comunque I_world per il momento angolare L
        I_world = R @ self.I @ R.T
        L = I_world @ omega

        
        u_sum = sum(u[i*3 : (i+1)*3] for i in range(8))
        return cs.vertcat(
            x[3:6], 
            (1 / self.mass) * u_sum + self.g, 
            self.compute_quaternion_derivative(q, omega),
            I_world_inv @ (self.compute_total_torque(x, u, p_contacts) - cs.cross(omega, L))
        )

    def quat_to_rot(self, q):
        """Converte un quaternione [w, x, y, z] in matrice di rotazione 3x3 usando CasADi."""
        w, x, y, z = q[0], q[1], q[2], q[3]
        
        R = cs.vertcat(
            cs.horzcat(1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w),
            cs.horzcat(2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w),
            cs.horzcat(2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2)
        )
        return R  
          
    def compute_quaternion_derivative(self, q, w):
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        wx, wy, wz = w[0], w[1], w[2]
        E = cs.vertcat(
            cs.horzcat(-qx, -qy, -qz),
            cs.horzcat(qw, -qz, qy),
            cs.horzcat(qz, qw, -qx),
            cs.horzcat(-qy, qx, qw)
        )
        return 0.5 * (E @ w)

    def compute_total_torque(self, x, u, p_contacts):
        com_pos = x[0:3]
        total_torque = cs.vertcat(0.0, 0.0, 0.0)
        
        for i in range(8):
            f_i = u[i*3 : (i+1)*3]
            p_i = p_contacts[i*3 : (i+1)*3]
            
            # lever length
            r = p_i - com_pos
            
            # L_dot = r x f
            total_torque += cs.cross(r, f_i)
        return total_torque        

    def apply_kinematic_constraints(self, t):
        current_step_index = self.footstep_planner.get_step_index_at_time(t)
        support_foot_pos = self.footstep_planner.plan[current_step_index]['pos']
        support_id = self.footstep_planner.plan[current_step_index]['foot_id']
        
        L_max = 0.5 # maximum leg length

        # Limite radiale semplice
        self.opt.subject_to( 
            (self.p_swing[0] - support_foot_pos[0])**2 + 
            (self.p_swing[1] - support_foot_pos[1])**2 <= L_max**2 
        )
        
        # VINCOLO ANTI-COMPENETRAZIONE (No Crossed Legs)
        # La gamba sinistra (Y positivo) deve restare a sinistra della destra (Y negativo)
        min_clearance = 0.10  # Minimo 10 cm tra i piedi
        if support_id == 'lfoot':
            # Piede d'appoggio è il sinistro, quindi p_swing è il destro.
            # Il destro deve restare a DESTRA del sinistro (y_destro < y_sinistro - clearance)
            self.opt.subject_to(self.p_swing[1] <= support_foot_pos[1] - min_clearance)
        else:
            # Piede d'appoggio è il destro, quindi p_swing è il sinistro.
            # Il sinistro deve restare a SINISTRA del destro (y_sinistro > y_destro + clearance)
            self.opt.subject_to(self.p_swing[1] >= support_foot_pos[1] + min_clearance)
        
    def compute_controls(self, current_state, t, nominal_plan=None):
        planner_tick = int(round(t / self.delta))
        current_step_index = self.footstep_planner.get_step_index_at_time(planner_tick)
        if current_step_index is None:
            current_step_index = len(self.footstep_planner.plan) - 1

        if nominal_plan is not None:
            plan_for_target = nominal_plan
        else:
            plan_for_target = self.footstep_planner.plan

        try:
            next_step_target = plan_for_target[current_step_index + 1]['pos'][0:2]
        except IndexError:
            next_step_target = plan_for_target[current_step_index]['pos'][0:2]

        if not self._is_finite_state(current_state):
            self.last_solve_success = False
            self.last_solve_message = "MPC skipped due to non-finite current state"
            optimal_controls, target_state = self._build_safe_fallback(current_state, next_step_target)
            phase_now = self.footstep_planner.get_phase_at_time(planner_tick)
            if phase_now == 'ds':
                contact = 'ds'
            else:
                step_idx = self.footstep_planner.get_step_index_at_time(planner_tick)
                contact = self.footstep_planner.plan[step_idx]['foot_id']
            return optimal_controls, target_state, contact, self.last_p_swing

        #  Inizializzazione Problema di Ottimizzazione
        self.opt = cs.Opti()
        self.X = self.opt.variable(13, self.N + 1) 
        self.U = self.opt.variable(24, self.N)     
        self.p_swing = self.opt.variable(2)

        # INIZIALIZZAZIONE E WARM START 
        if getattr(self, 'last_X', None) is not None:
            # Shift buffer di 5 ticks (frequenza simulatore 100Hz / MPC 20Hz)
            shift = 5
            if self.N > shift:
                X_guess = np.hstack((self.last_X[:, shift:], np.tile(self.last_X[:, -1:], (1, shift))))
                U_guess = np.hstack((self.last_U[:, shift:], np.tile(self.last_U[:, -1:], (1, shift))))
            else:
                X_guess = self.last_X
                U_guess = self.last_U
            self.opt.set_initial(self.X, X_guess)
            self.opt.set_initial(self.U, U_guess)
        else:
            for k in range(self.N + 1):
                self.opt.set_initial(self.X[0:3, k], current_state['com']['pos'])
                self.opt.set_initial(self.X[6:10, k], current_state['base']['quat'])
            
            f_z_guess = (self.params['mass'] * 9.81) / 8.0
            for i in range(8):
                self.opt.set_initial(self.U[i*3 + 2, :], f_z_guess)        
        
        p_opts = {"expand": True, "print_time": False}
        s_opts = {
            "max_iter": 500,
            "print_level": 0,
            "sb": "yes",
            "tol": 1e-3 
        }
        self.opt.solver('ipopt', p_opts, s_opts)
        
        # Set initial state
        x0 = cs.vertcat(
            current_state['com']['pos'],
            current_state['com']['vel'],
            current_state['base']['quat'],
            current_state['base']['omega']
        )
        self.opt.subject_to(self.X[:, 0] == x0)
        
        current_support = self.footstep_planner.plan[current_step_index]['foot_id']
        
        #  TUNING PESI 
        cost = 0.0
        W_com_z = 5000
        W_com_xy = 100
        W_vel = np.diag([1.0, 1.0, 5.0])
        W_quat = np.diag([15.0, 15.0, 100.0])
        W_omega = np.diag([3.0, 3.0, 3.0])
        W_force = np.eye(24) * 1e-6
        W_swing = 500.0
        W_quat_norm = 100.0

        h_ref = self.initial['com']['pos'][2] 
        
        # --- MAIN LOOP ---
        for k in range(self.N): 
            t_k = t + k * self.delta
            planner_tick_k = planner_tick + k
            future_step_index = self.footstep_planner.get_step_index_at_time(planner_tick_k)
            phase = self.footstep_planner.get_phase_at_time(planner_tick_k)
            support_foot = self.footstep_planner.plan[future_step_index]['foot_id']
            
            # --- UNIVERSAL HORIZON EVALUATOR ---
            if nominal_plan is not None:
                plan_to_use = nominal_plan
            else:
                plan_to_use = self.footstep_planner.plan

            # Trova l'ultimo step in cui LFOOT era il piede di swing
            last_swing_l = -1
            for j in range(current_step_index, future_step_index + 1):
                j_valid = min(j, len(plan_to_use)-1)
                # se il supporto è rfoot, LFOOT è lo swing
                if plan_to_use[j_valid]['foot_id'] == 'rfoot': 
                    last_swing_l = j_valid
            
            if last_swing_l == -1:
                p_lfoot_k = current_state['lfoot']['pos'][3:5]
                yaw_l_k   = current_state['lfoot']['pos'][2]
            elif last_swing_l == current_step_index:
                p_lfoot_k = self.p_swing
                yaw_l_k   = plan_to_use[last_swing_l]['ang'][2]
            else:
                p_lfoot_k = plan_to_use[last_swing_l]['pos'][0:2]
                yaw_l_k   = plan_to_use[last_swing_l]['ang'][2]

            # Trova l'ultimo step in cui RFOOT era il piede di swing
            last_swing_r = -1
            for j in range(current_step_index, future_step_index + 1):
                j_valid = min(j, len(plan_to_use)-1)
                # se il supporto è lfoot, RFOOT è lo swing
                if plan_to_use[j_valid]['foot_id'] == 'lfoot': 
                    last_swing_r = j_valid

            if last_swing_r == -1:
                p_rfoot_k = current_state['rfoot']['pos'][3:5]
                yaw_r_k   = current_state['rfoot']['pos'][2]
            elif last_swing_r == current_step_index:
                p_rfoot_k = self.p_swing
                yaw_r_k   = plan_to_use[last_swing_r]['ang'][2]
            else:
                p_rfoot_k = plan_to_use[last_swing_r]['pos'][0:2]
                yaw_r_k   = plan_to_use[last_swing_r]['ang'][2]
            # --- END UNIVERSAL HORIZON EVALUATOR ---

            p_contacts = self.generate_contact_points(p_lfoot_k, p_rfoot_k, yaw_l_k, yaw_r_k, 0.0)

            # DINAMICA E VINCOLI FISICI 
            x_k = self.X[:, k]
            u_k = self.U[:, k]
            
            # Integrazione dinamica 
            x_next = x_k + self.delta * self.f(x_k, u_k, p_contacts)
            self.opt.subject_to(self.X[:, k + 1] == x_next)
            
            
            
            for i in range(8):
                fx = self.U[i*3 + 0, k]
                fy = self.U[i*3 + 1, k]
                fz = self.U[i*3 + 2, k]
                
                # Forza Z minima a 0.1 per stabilità numerica
                self.opt.subject_to(self.opt.bounded(0.0, fz, 500.0)) 
                self.opt.subject_to(self.opt.bounded(-self.mu * fz, fx, self.mu * fz))
                self.opt.subject_to(self.opt.bounded(-self.mu * fz, fy, self.mu * fz))
            
            step_idx_k = self.footstep_planner.get_step_index_at_time(planner_tick_k)
            support_foot_k = self.footstep_planner.plan[step_idx_k]['foot_id']
            swing_foot_k = 'lfoot' if support_foot_k == 'rfoot' else 'rfoot'

            if phase == 'ss':
                if swing_foot_k == 'lfoot':
                    self.opt.subject_to(self.opt.bounded(-1e-4, self.U[0:12, k], 1e-4))
                else:
                    self.opt.subject_to(self.opt.bounded(-1e-4, self.U[12:24, k], 1e-4))

            
            # Cost function
            # Altezza CoM
            cost += W_com_z * (self.X[2, k + 1] - h_ref)**2
            
            # Target XY del CoM (Spostamento del peso)
            if phase == 'ds':
                com_xy_target = (p_lfoot_k + p_rfoot_k) / 2.0
            else:
                if swing_foot_k == 'lfoot':
                    # Supporto DESTRO: spostiamo solo 2cm in avanti, ZERO lateralmente
                    com_xy_target = p_rfoot_k + np.array([0.02, 0.0]) 
                else:
                    # Supporto SINISTRO: spostiamo solo 2cm in avanti, ZERO lateralmente
                    com_xy_target = p_lfoot_k + np.array([0.02, 0.0])
            
            cost += W_com_xy * cs.sumsqr(self.X[0:2, k+1] - com_xy_target)
            
            # Regolarizzazioni (Velocità, Orientamento, Omega)
            cost += cs.mtimes([(self.X[3:6, k+1]).T, W_vel, self.X[3:6, k+1]])
            
            # Tracking dell'orientamento Yaw dinamico lungo la traiettoria
            yaw_ref = (yaw_l_k + yaw_r_k) / 2.0
            q_ref_w = cs.cos(yaw_ref / 2.0)
            q_ref_z = cs.sin(yaw_ref / 2.0)
            
            # Penalizzazioni indipendenti: Roll e Pitch (fermi), Yaw (segue le orme)
            cost += W_quat[0,0] * self.X[7, k+1]**2
            cost += W_quat[1,1] * self.X[8, k+1]**2
            q_dot_err = self.X[6, k+1]*q_ref_w + self.X[9, k+1]*q_ref_z
            cost += W_quat[2,2] * (1.0 - q_dot_err**2)
            
            cost += cs.mtimes([(self.X[10:13, k+1]).T, W_omega, self.X[10:13, k+1]])
            cost += W_quat_norm * (cs.sumsqr(self.X[6:10, k+1]) - 1.0)**2
            cost += cs.mtimes([u_k.T, W_force, u_k])
                
        # Vincoli cinematici gamba
        self.apply_kinematic_constraints(planner_tick)
        
        # Target per il piede che atterrerà (p_swing)
        cost += W_swing * cs.sumsqr(self.p_swing - next_step_target)
        
        self.opt.minimize(cost)
        
        try:
            sol = self.opt.solve()
            self.last_solve_success = True
            self.last_solve_message = ""
            
            # Salvare per Warm Start e Buffering
            self.last_X = sol.value(self.X)
            self.last_U = sol.value(self.U)
            self.last_p_swing = sol.value(self.p_swing)
            
            optimal_controls = self.last_U[:, 0]
            target_state = self.extract_target_state(sol)
            print(f"--- STEP {t} --- [V] IPOPT | Iters: {self.opt.stats()['iter_count']}")
        except Exception as e:
            self.last_solve_success = False
            self.last_solve_message = str(e)
            print(f"--- STEP {t} --- [X] IPOPT FALLITO!")
            try:
                self.opt.debug.show_infeasibilities()
            except Exception:
                pass

            optimal_controls, target_state = self._build_safe_fallback(current_state, next_step_target)
        
        fz_tot = sum(optimal_controls[i*3 + 2] for i in range(8))
        print(f"Forza Z Totale: {fz_tot:.1f} N | Coppia Max: {np.max(np.abs(optimal_controls)):.1f}")

        phase_now = self.footstep_planner.get_phase_at_time(planner_tick)
        if phase_now == 'ds':
            contact = 'ds'
        else:
            step_idx = self.footstep_planner.get_step_index_at_time(planner_tick)
            contact = self.footstep_planner.plan[step_idx]['foot_id']
            
        return optimal_controls, target_state, contact, self.last_p_swing
    
    
    def get_buffered_forces(self, tick_offset):
        if not hasattr(self, 'last_U') or self.last_U is None:
            return np.zeros(24)
        idx = min(tick_offset, self.N - 1)
        return self.last_U[:, idx]

    def get_buffered_state(self, tick_offset):
        if not hasattr(self, 'last_X') or self.last_X is None:
            return None
        idx = min(tick_offset + 1, self.N)
        x_target = self.last_X[:, idx]
        return {
            'com': {
                'pos': x_target[0:3],
                'vel': x_target[3:6],
                'acc': np.zeros(3) 
            },
            'base': {
                'quat': x_target[6:10],
                'omega': x_target[10:13]
            }
        }
    
    def generate_contact_points(self, p_left_xy, p_right_xy, yaw_l, yaw_r, z):
        d = self.foot_size / 2.0
        pts = []
        
        # Rotate offsets by yaw_l
        cos_l, sin_l = cs.cos(yaw_l), cs.sin(yaw_l)
        for x_sign in [1, -1]:
            for y_sign in [1, -1]:
                dx = x_sign * d
                dy = y_sign * d
                rx = dx * cos_l - dy * sin_l
                ry = dx * sin_l + dy * cos_l
                pts.append(cs.vertcat(p_left_xy[0] + rx, p_left_xy[1] + ry, z))
                
        # Rotate offsets by yaw_r
        cos_r, sin_r = cs.cos(yaw_r), cs.sin(yaw_r)
        for x_sign in [1, -1]:
            for y_sign in [1, -1]:
                dx = x_sign * d
                dy = y_sign * d
                rx = dx * cos_r - dy * sin_r
                ry = dx * sin_r + dy * cos_r
                pts.append(cs.vertcat(p_right_xy[0] + rx, p_right_xy[1] + ry, z))
        return cs.vertcat(*pts)
    
    def extract_target_state(self, sol):
        x_target = sol.value(self.X[:, 1])
        return {
            'com': {
                'pos': x_target[0:3],
                'vel': x_target[3:6],
                'acc': np.zeros(3) 
            },
            'base': {
                'quat': x_target[6:10],
                'omega': x_target[10:13]
            }
        }
