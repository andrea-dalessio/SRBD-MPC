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

    def _get_dynamics_with_rot_inertia(self, x, u, p_contacts):
        q = x[6:10]      
        omega = x[10:13] 
        
        R = self.quat_to_rot(q)
        
        # --- NUOVA LOGICA SENZA cs.inv() ---
        # Ruotiamo l'inversa invece di invertire la matrice ruotata
        I_world_inv = R @ self.I_body_inv @ R.T
        
        # Ci serve comunque I_world per il momento angolare L
        I_world = R @ self.I @ R.T
        L = I_world @ omega
        # -----------------------------------
        
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
        
        L_max = 0.5 # maximum leg length

        self.opt.subject_to( 
            (self.p_swing[0] - support_foot_pos[0])**2 + 
            (self.p_swing[1] - support_foot_pos[1])**2 <= L_max**2 
        )
        
    def compute_controls(self, current_state, t):
        planner_tick = int(round(t / self.delta))
        # 3. Inizializzazione Problema di Ottimizzazione
        self.opt = cs.Opti()
        self.X = self.opt.variable(13, self.N + 1) 
        self.U = self.opt.variable(24, self.N)     
        self.p_swing = self.opt.variable(2)

        # --- INIZIALIZZAZIONE ---
        for k in range(self.N + 1):
            self.opt.set_initial(self.X[0:3, k], current_state['com']['pos'])
            self.opt.set_initial(self.X[6:10, k], current_state['base']['quat'])
        
        # Guess iniziale delle forze pari a un ottavo del peso per ogni punto di contatto
        f_z_guess = (self.params['mass'] * 9.81) / 8.0
        for i in range(8):
            self.opt.set_initial(self.U[i*3 + 2, :], f_z_guess)        
        
        p_opts = {"expand": True, "print_time": False}
        s_opts = {
            "max_iter": 500,
            "print_level": 0,
            "sb": "yes",
            "tol": 1e-3 # Tolleranza leggermente più alta per favorire la velocità
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
        
        current_step_index = self.footstep_planner.get_step_index_at_time(planner_tick)
        current_support = self.footstep_planner.plan[current_step_index]['foot_id']
        
        # --- TUNING PESI (Aggiornati per favorire il moto) ---
        cost = 0.0
        W_com_z = 5000
        W_com_xy = 100
        W_vel = np.diag([1.0, 1.0, 5.0])
        W_quat = np.diag([20.0, 20.0, 20.0])
        W_omega = np.diag([5.0, 5.0, 5.0])
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
            
            # Determinazione posizioni piedi nell'orizzonte
            if future_step_index == current_step_index: 
                p_lfoot_k = current_state['lfoot']['pos'][3:5]
                yaw_l_k   = current_state['lfoot']['pos'][2]
                p_rfoot_k = current_state['rfoot']['pos'][3:5]
                yaw_r_k   = current_state['rfoot']['pos'][2]
            else: 
                if current_support == 'lfoot':
                    p_lfoot_k = current_state['lfoot']['pos'][3:5]
                    yaw_l_k   = current_state['lfoot']['pos'][2]
                    p_rfoot_k = self.p_swing # Piede destro è quello che atterrerà
                    
                    next_idx = min(future_step_index, len(self.footstep_planner.plan)-1)
                    yaw_r_k   = self.footstep_planner.plan[next_idx]['ang'][2]
                else:
                    p_lfoot_k = self.p_swing # Piede sinistro è quello che atterrerà
                    next_idx = min(future_step_index, len(self.footstep_planner.plan)-1)
                    yaw_l_k   = self.footstep_planner.plan[next_idx]['ang'][2]
                    p_rfoot_k = current_state['rfoot']['pos'][3:5]
                    yaw_r_k   = current_state['rfoot']['pos'][2]

            p_contacts = self.generate_contact_points(p_lfoot_k, p_rfoot_k, yaw_l_k, yaw_r_k, 0.0)

            # --- DINAMICA E VINCOLI FISICI ---
            x_k = self.X[:, k]
            u_k = self.U[:, k]
            
            # Integrazione dinamica (ora con inerzia ruotata gestita in self.f)
            x_next = x_k + self.delta * self.f(x_k, u_k, p_contacts)
            self.opt.subject_to(self.X[:, k + 1] == x_next)
            
            # Vincolo unitarietà del quaternione (fondamentale per SRBD)
            
            
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

            
            # --- FUNZIONE DI COSTO ---
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
            cost += cs.mtimes([(self.X[7:10, k+1]).T, W_quat, self.X[7:10, k+1]])
            cost += cs.mtimes([(self.X[10:13, k+1]).T, W_omega, self.X[10:13, k+1]])
            cost += W_quat_norm * (cs.sumsqr(self.X[6:10, k+1]) - 1.0)**2
            cost += cs.mtimes([u_k.T, W_force, u_k])
                
        # Vincoli cinematici gamba
        self.apply_kinematic_constraints(planner_tick)
        
        # Target per il piede che atterrerà (p_swing)
        try:
            next_step_target = self.footstep_planner.plan[current_step_index + 1]['pos'][0:2]
        except IndexError:
            next_step_target = self.footstep_planner.plan[current_step_index]['pos'][0:2]  
        cost += W_swing * cs.sumsqr(self.p_swing - next_step_target)
        
        self.opt.minimize(cost)
        
        # --- SOLUZIONE ---
        try:
            sol = self.opt.solve()
            optimal_controls = sol.value(self.U[:, 0])
            target_state = self.extract_target_state(sol)
            print(f"--- STEP {t} --- [V] IPOPT Successo")
        except Exception as e:
            print(f"--- STEP {t} --- [X] IPOPT FALLITO!")
            self.opt.debug.show_infeasibilities()

            # Fallback sicuro
            optimal_controls = np.zeros(24)
            fz_each = (self.mass * abs(self.g[2])) / 8.0
            for i in range(8):
                optimal_controls[i*3 + 2] = fz_each

            # Target conservativo: resta sullo stato corrente
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
        
        fz_tot = sum(optimal_controls[i*3 + 2] for i in range(8))
        print(f"Forza Z Totale: {fz_tot:.1f} N | Coppia Max: {np.max(np.abs(optimal_controls)):.1f}")

        phase_now = self.footstep_planner.get_phase_at_time(planner_tick)
        if phase_now == 'ds':
            contact = 'ds'
        else:
            step_idx = self.footstep_planner.get_step_index_at_time(planner_tick)
            contact = self.footstep_planner.plan[step_idx]['foot_id']
            
        return optimal_controls, target_state, contact
    
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
