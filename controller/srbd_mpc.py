import numpy as np
import casadi as cs
import os

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
        self.debug_verbose = os.environ.get('MPC_DEBUG_VERBOSE', '0').strip().lower() in ['1', 'true', 'yes']
        self.no_replan_tolerance_m = float(os.environ.get('MPC_NO_REPLAN_TOL_M', '0.10'))
        self.max_step_length_m = float(os.environ.get('MPC_MAX_STEP_LENGTH_M', '0.35'))
        self.min_lateral_clearance_m = float(os.environ.get('MPC_MIN_LATERAL_CLEARANCE_M', '0.05'))
        self.fz_sum_min_factor_ds = float(os.environ.get('MPC_FZ_SUM_MIN_FACTOR_DS', '0.70'))
        self.fz_sum_max_factor_ds = float(os.environ.get('MPC_FZ_SUM_MAX_FACTOR_DS', '1.35'))
        self.fz_sum_min_factor_ss = float(os.environ.get('MPC_FZ_SUM_MIN_FACTOR_SS', '0.60'))
        self.fz_sum_max_factor_ss = float(os.environ.get('MPC_FZ_SUM_MAX_FACTOR_SS', '1.50'))
        
        # Dynamics
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

    def _build_safe_fallback(self, current_state, next_step_target, phase='ds', support_foot='ds'):
        optimal_controls = np.zeros(24)
        total_weight = self.mass * abs(self.g[2])

        # Contact-aware fallback: distribute only on active contacts to avoid
        # unrealistic vertical transients when MPC fails during single support.
        if phase == 'ss' and support_foot in ['lfoot', 'rfoot']:
            if support_foot == 'lfoot':
                active_contacts = range(0, 4)
            else:
                active_contacts = range(4, 8)
            fz_each = total_weight / 4.0
        else:
            active_contacts = range(0, 8)
            fz_each = total_weight / 8.0

        # Slightly reduce fallback vertical force when rising quickly,
        # to prevent jump-like behavior after failure.
        com_vz = float(current_state['com']['vel'][2])
        if com_vz > 0.12:
            fz_each *= 0.9

        for i in active_contacts:
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
            for i in active_contacts:
                self.last_U[i*3 + 2, k] = fz_each

        # Compute a capture-point corrective step target during ss fallback.
        # When IPOPT fails, the nominal next_step_target has zero disturbance
        # correction — so footstep replanning in the simulation would use a
        # step that makes no attempt to recover. Instead, compute:
        #   p_capture = p_com + (1/omega) * v_lateral
        # where omega = sqrt(g / h_com) from the LIPM, then blend it with the
        # nominal target so we don't step too far from the kinematic workspace.
        if phase == 'ss':
            try:
                g_mag = abs(self.g[2])
                h_com = float(current_state['com']['pos'][2])
                if h_com > 0.05:
                    omega = float(np.sqrt(g_mag / h_com))
                    com_xy  = np.array(current_state['com']['pos'][0:2], dtype=float)
                    vel_xy  = np.array(current_state['com']['vel'][0:2], dtype=float)
                    capture = com_xy + vel_xy / omega          # classic capture point
                    nominal = np.array(next_step_target[0:2],  dtype=float)
                    # Blend: 60% capture correction, 40% nominal to stay in workspace
                    corrected = 0.60 * capture + 0.40 * nominal
                    p_swing = np.array(next_step_target, dtype=float)
                    p_swing[0] = corrected[0]
                    p_swing[1] = corrected[1]
                    self.last_p_swing = p_swing
                else:
                    self.last_p_swing = np.array(next_step_target).copy()
            except Exception:
                self.last_p_swing = np.array(next_step_target).copy()
        else:
            self.last_p_swing = np.array(next_step_target).copy()
        return optimal_controls, target_state

    def _get_dynamics_with_rot_inertia(self, x, u, p_contacts):
        q = x[6:10]      
        omega = x[10:13] 
        
        R = self.quat_to_rot(q)
        

        I_world_inv = R @ self.I_body_inv @ R.T
        
        # Compute inertia in world frame to get angular momentum
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
            cs.horzcat(qw, qz, -qy),
            cs.horzcat(-qz, qw, qx),
            cs.horzcat(qy, -qx, qw)
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
        
        # HRP-4 physical dimensions (from URDF):
        #   Foot collision box : 0.25 m (x) x 0.13 m (y)
        #   Hip lateral offset : ±0.0875 m  →  nominal foot centre-to-centre = 0.175 m
        #
        # Soft lateral-separation constraint:
        #   We enforce a minimum lateral clearance between the swing foot target and the
        #   support foot centre using a SOFT quadratic penalty rather than a hard constraint.
        #   This keeps IPOPT feasible under large disturbances (hard constraints would cause
        #   infeasibility → solver failure) while still strongly discouraging foot overlap.
        #
        #   Minimum safe gap = half foot width (0.065 m) + safety margin (0.05 m) = 0.115 m
        #   This equals roughly the HRP-4 nominal hip-to-foot lateral distance, so during
        #   normal walking the penalty is near-zero; it only activates when the MPC tries
        #   to plan a dangerously narrow or crossing step.
        #
        #   W_lateral_sep scales the softness:
        #     • Normal walking  → clearance comfortably satisfied → penalty ≈ 0
        #     • Recovery push   → allowed to shrink toward 0.05 m under extreme disturbance
        #     • Cross-over step → penalty rises steeply, discouraging foot collision

        # HRP-4 half foot width + safety margin (metres)
        HRP4_FOOT_HALF_WIDTH = 0.065   # half of 0.13 m foot y-dimension
        HRP4_SAFETY_MARGIN   = 0.05    # additional clearance buffer
        min_lateral_sep = HRP4_FOOT_HALF_WIDTH + HRP4_SAFETY_MARGIN   # 0.115 m

        # Soft penalty weight — high enough to keep normal steps clear,
        # low enough that IPOPT stays feasible under severe disturbance.
        W_lateral_sep = 3000.0

        support_foot_y = support_foot_pos[1]   # world-frame Y of support foot centre

        # Determine which side the swing foot should be on:
        #   if support is lfoot → swing (rfoot) should be to the RIGHT  (y < support_y)
        #   if support is rfoot → swing (lfoot) should be to the LEFT   (y > support_y)
        if support_id == 'lfoot':
            # rfoot swinging → penalise if p_swing[1] > support_y - min_lateral_sep
            violation = cs.fmax(0.0, self.p_swing[1] - (support_foot_y - min_lateral_sep))
        else:
            # lfoot swinging → penalise if p_swing[1] < support_y + min_lateral_sep
            violation = cs.fmax(0.0, (support_foot_y + min_lateral_sep) - self.p_swing[1])

        # Return the soft penalty so compute_controls() can add it to cost before minimize()
        return W_lateral_sep * violation**2
        
    def compute_controls(self, current_state, t, nominal_plan=None, allow_footstep_replanning=True):
        planner_tick = int(round(t / self.delta))
        current_step_index = self.footstep_planner.get_step_index_at_time(planner_tick)
        if current_step_index is None:
            current_step_index = len(self.footstep_planner.plan) - 1

        if nominal_plan is not None:
            plan_for_target = nominal_plan
        else:
            plan_for_target = self.footstep_planner.plan

        support_id_now = self.footstep_planner.plan[current_step_index]['foot_id']
        swing_id_now = 'lfoot' if support_id_now == 'rfoot' else 'rfoot'

        has_future_step = (current_step_index + 1) < len(plan_for_target)

        if has_future_step:
            next_step_target = plan_for_target[current_step_index + 1]['pos'][0:2]
        else:
            # End-of-plan fallback: keep swing target near the current swing-foot position.
            # This avoids conflicting constraints when no landing step is available.
            next_step_target = np.array(current_state[swing_id_now]['pos'][3:5], dtype=float)

        if not self._is_finite_state(current_state):
            self.last_solve_success = False
            self.last_solve_message = "MPC skipped due to non-finite current state"
            phase_now = self.footstep_planner.get_phase_at_time(planner_tick)
            if phase_now == 'ss':
                support_for_fallback = self.footstep_planner.plan[current_step_index]['foot_id']
            else:
                support_for_fallback = 'ds'

            optimal_controls, target_state = self._build_safe_fallback(
                current_state,
                next_step_target,
                phase=phase_now,
                support_foot=support_for_fallback
            )
            if phase_now == 'ds':
                contact = 'ds'
            else:
                step_idx = self.footstep_planner.get_step_index_at_time(planner_tick)
                contact = self.footstep_planner.plan[step_idx]['foot_id']
            return optimal_controls, target_state, contact, self.last_p_swing

        # Optimization problem setup
        self.opt = cs.Opti()
        self.X = self.opt.variable(13, self.N + 1) 
        self.U = self.opt.variable(24, self.N)     
        self.p_swing = self.opt.variable(2)
 
        if getattr(self, 'last_X', None) is not None:
            # Shift buffer of 5 ticks (simulator frequency 100Hz / MPC 20Hz) to warm start the MPC
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
            # The rotational dynamics (R @ I_inv @ R.T) create a nonlinear NLP
            # that needs enough iterations to converge, especially on cold starts.
            # 600 gives recovery solves enough budget; warm starts exit via acceptable_iter.
            "max_iter": 600,
            "print_level": 0,
            "sb": "yes",
            "tol": 1e-3,
            # Acceptable early exit: on warm-started steps IPOPT often reaches
            # an acceptable solution in 10-30 iters and exits without hitting max_iter.
            "acceptable_tol": 5e-3,
            "acceptable_iter": 3,
            # Faster linear algebra via MUMPS
            "mumps_pivtol": 1e-4,
            # Warm-start hints
            "warm_start_init_point": "yes",
            "warm_start_bound_push": 1e-6,
            "warm_start_mult_bound_push": 1e-6,
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
        
        # Weights 
        cost = 0.0
        W_com_z = 5000.0
        W_com_x = 100.0
        W_com_y = 20.0
        W_vel = np.diag([1.0, 1.0, 5.0])
        if allow_footstep_replanning:
            W_quat = np.diag([200.0, 200.0, 20.0])
            W_omega = np.diag([10.0, 10.0, 5.0])
        else:
            W_quat = np.diag([200.0, 200.0, 200.0])
            W_omega = np.diag([10.0, 10.0, 10.0])
        W_force = np.eye(24) * 1e-6
        W_swing = 500.0
        W_quat_norm = 100.0

        h_ref = self.initial['com']['pos'][2] 
        
        # Main loop
        for k in range(self.N): 
            t_k = t + k * self.delta
            planner_tick_k = planner_tick + k
            future_step_index = self.footstep_planner.get_step_index_at_time(planner_tick_k)
            phase = self.footstep_planner.get_phase_at_time(planner_tick_k)
            support_foot = self.footstep_planner.plan[future_step_index]['foot_id']
            
            # Horizon evaluator (Use the nominal plan if replanner hasn't triggered)
            if nominal_plan is not None:
                plan_to_use = nominal_plan
            else:
                plan_to_use = self.footstep_planner.plan

            # Find the last step in which LFOOT was the swing foot
            last_swing_l = -1
            for j in range(current_step_index, future_step_index + 1):
                j_valid = min(j, len(plan_to_use)-1)
                # if support is rfoot, LFOOT is the swing foot
                if plan_to_use[j_valid]['foot_id'] == 'rfoot': 
                    last_swing_l = j_valid
            
            # If no swing foot is found, use the current state
            if last_swing_l == -1:
                p_lfoot_k = current_state['lfoot']['pos'][3:5]
                yaw_l_k   = current_state['lfoot']['pos'][2]
            elif last_swing_l == current_step_index:
                p_lfoot_k = self.p_swing
                yaw_l_k   = plan_to_use[last_swing_l]['ang'][2]
            else:
                p_lfoot_k = plan_to_use[last_swing_l]['pos'][0:2]
                yaw_l_k   = plan_to_use[last_swing_l]['ang'][2]

            # Find the last step in which RFOOT was the swing foot
            last_swing_r = -1
            for j in range(current_step_index, future_step_index + 1):
                j_valid = min(j, len(plan_to_use)-1)
                # if support is lfoot, RFOOT is the swing foot
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

            p_contacts = self.generate_contact_points(p_lfoot_k, p_rfoot_k, yaw_l_k, yaw_r_k, 0.0)

            # Dynamics and physical constraints
            x_k = self.X[:, k]
            u_k = self.U[:, k]
            
            # Dynamic integration (Euler)
            x_next = x_k + self.delta * self.f(x_k, u_k, p_contacts)
            self.opt.subject_to(self.X[:, k + 1] == x_next)
            
            
            
            for i in range(8):
                fx = self.U[i*3 + 0, k]
                fy = self.U[i*3 + 1, k]
                fz = self.U[i*3 + 2, k]
                
                # Minimum Z force for numerical stability
                self.opt.subject_to(self.opt.bounded(0.0, fz, 500.0)) 
                self.opt.subject_to(self.opt.bounded(-self.mu * fz, fx, self.mu * fz))
                self.opt.subject_to(self.opt.bounded(-self.mu * fz, fy, self.mu * fz))

            # Hard bound on angular velocity to prevent numerical explosion in
            # the rotational dynamics (R @ I_inv @ R.T). Without this, IPOPT can
            # produce omega > 100 rad/s when it hasn't converged, causing the
            # fallback path to trigger repeatedly and the robot to fall.
            self.opt.subject_to(self.opt.bounded(-8.0, self.X[10, k+1],  8.0))  # omega_x
            self.opt.subject_to(self.opt.bounded(-8.0, self.X[11, k+1],  8.0))  # omega_y
            self.opt.subject_to(self.opt.bounded(-4.0, self.X[12, k+1],  4.0))  # omega_z (yaw rate)

            # Limit total vertical force to avoid jump-like transients when MPC is stressed.
            fz_sum = 0.0
            for i in range(8):
                fz_sum += self.U[i*3 + 2, k]

            mg = self.mass * abs(self.g[2])
            if phase == 'ss':
                fz_min = self.fz_sum_min_factor_ss * mg
                fz_max = self.fz_sum_max_factor_ss * mg
            else:
                fz_min = self.fz_sum_min_factor_ds * mg
                fz_max = self.fz_sum_max_factor_ds * mg
            self.opt.subject_to(self.opt.bounded(fz_min, fz_sum, fz_max))
            
            step_idx_k = self.footstep_planner.get_step_index_at_time(planner_tick_k)
            support_foot_k = self.footstep_planner.plan[step_idx_k]['foot_id']
            swing_foot_k = 'lfoot' if support_foot_k == 'rfoot' else 'rfoot'

            if phase == 'ss':
                if swing_foot_k == 'lfoot':
                    self.opt.subject_to(self.opt.bounded(-1e-4, self.U[0:12, k], 1e-4))
                else:
                    self.opt.subject_to(self.opt.bounded(-1e-4, self.U[12:24, k], 1e-4))

            
            # COST FUNCTIONS
            # COM Height
            cost += W_com_z * (self.X[2, k + 1] - h_ref)**2
            
            # CoM Target XY
            # Move smoothly towards the support foot to cancel gravity
            if phase == 'ds':
                com_xy_target = (p_lfoot_k + p_rfoot_k) / 2.0
            else:
                if swing_foot_k == 'lfoot':
                    com_xy_target = p_rfoot_k + np.array([0.02, 0.0])
                else:
                    com_xy_target = p_lfoot_k + np.array([0.02, 0.0])
            
            cost += W_com_x * (self.X[0, k+1] - com_xy_target[0])**2
            cost += W_com_y * (self.X[1, k+1] - com_xy_target[1])**2
            
            # Regularization (Velocity, Orientation, Omega)
            cost += cs.mtimes([(self.X[3:6, k+1]).T, W_vel, self.X[3:6, k+1]])
            
            # Dynamic yaw orientation tracking along the trajectory
            yaw_ref = (yaw_l_k + yaw_r_k) / 2.0
            q_ref_w = cs.cos(yaw_ref / 2.0)
            q_ref_z = cs.sin(yaw_ref / 2.0)
            
            # Independent penalties: Roll and Pitch (fixed), Yaw (tracks the footsteps)
            # Calculate quaternion error q_err = q_ref* * q
            # q_err_x and q_err_y represent the roll and pitch error in the body frame
            # q_err_z represents the yaw error
            q_err_x = q_ref_w * self.X[7, k+1] - q_ref_z * self.X[8, k+1]
            q_err_y = q_ref_w * self.X[8, k+1] + q_ref_z * self.X[7, k+1]
            q_err_z = q_ref_w * self.X[9, k+1] - q_ref_z * self.X[6, k+1]
            
            cost += W_quat[0,0] * q_err_x**2
            cost += W_quat[1,1] * q_err_y**2
            cost += W_quat[2,2] * q_err_z**2
            
            cost += cs.mtimes([(self.X[10:13, k+1]).T, W_omega, self.X[10:13, k+1]])
            cost += W_quat_norm * (cs.sumsqr(self.X[6:10, k+1]) - 1.0)**2
            cost += cs.mtimes([u_k.T, W_force, u_k])
                
        # Leg and soft lateral separation constraint (HRP-4 foot geometry)
        lateral_sep_cost = self.apply_kinematic_constraints(planner_tick)
        cost += lateral_sep_cost

        lock_to_nominal = (not allow_footstep_replanning) and has_future_step

        if allow_footstep_replanning:
            # Target for the foot that will land (p_swing)
            cost += W_swing * cs.sumsqr(self.p_swing - next_step_target)
        elif lock_to_nominal:
            # Keep MPC and executed foot trajectory aligned in nominal-mode runs.
            cost += (10.0 * W_swing) * cs.sumsqr(self.p_swing - next_step_target)
        else:
            # No future step available: keep swing target close without hard locking.
            cost += W_swing * cs.sumsqr(self.p_swing - next_step_target)
        
        self.opt.minimize(cost)
        
        try:
            sol = self.opt.solve()
            self.last_solve_success = True
            self.last_solve_message = ""
            
            # Save for Warm Start and Buffering
            self.last_X = sol.value(self.X)
            self.last_U = sol.value(self.U)
            self.last_p_swing = sol.value(self.p_swing)
            
            optimal_controls = self.last_U[:, 0]
            target_state = self.extract_target_state(sol)
            if self.debug_verbose:
                print(f"--- STEP {t} --- [V] IPOPT | Iters: {self.opt.stats()['iter_count']}")
        except Exception as e:
            self.last_solve_success = False
            self.last_solve_message = str(e)
            if self.debug_verbose:
                print(f"--- STEP {t} --- [X] IPOPT FALLITO!")
            try:
                self.opt.debug.show_infeasibilities()
            except Exception:
                pass

            phase_now = self.footstep_planner.get_phase_at_time(planner_tick)
            if phase_now == 'ss':
                support_for_fallback = self.footstep_planner.plan[current_step_index]['foot_id']
            else:
                support_for_fallback = 'ds'
            optimal_controls, target_state = self._build_safe_fallback(
                current_state,
                next_step_target,
                phase=phase_now,
                support_foot=support_for_fallback
            )
        
        fz_tot = sum(optimal_controls[i*3 + 2] for i in range(8))
        if self.debug_verbose:
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
