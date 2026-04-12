import dartpy as dart
import numpy as np
import os
from utils import *

class InverseDynamics:
    def __init__(
        self,
        robot,
        redundant_dofs,
        foot_size=0.1,
        µ=0.5,
        control_dt=0.01,
        protected_joint_deviation_deg=None,
        enable_protected_joint_constraints=False,
        use_model_torque_limits=True,
        enable_knee_safety=True,
        knee_min_support_deg=12.0,
        knee_min_ds_deg=8.0,
        knee_output_guard_margin_deg=4.0,
        knee_output_guard_kp=120.0,
        knee_output_guard_kd=8.0,
        knee_output_guard_max_torque=45.0,
        ankle_pitch_torque_scale=1.0,
        ankle_roll_torque_scale=1.0,
        profile_name='wbc'
    ):
        self.robot = robot
        self.dofs = self.robot.getNumDofs()
        self.d = foot_size / 2.
        self.µ = µ
        self.control_dt = control_dt
        self.profile_name = profile_name
        self.use_model_torque_limits = bool(use_model_torque_limits)
        self.enable_knee_safety = bool(enable_knee_safety)
        self.debug_verbose = os.environ.get('WBC_DEBUG_VERBOSE', '0').strip().lower() in ['1', 'true', 'yes']
        self.knee_output_guard_margin = np.deg2rad(knee_output_guard_margin_deg)
        self.knee_output_guard_kp = float(knee_output_guard_kp)
        self.knee_output_guard_kd = float(knee_output_guard_kd)
        self.knee_output_guard_max_torque = float(knee_output_guard_max_torque)
        self.ankle_pitch_torque_scale = float(ankle_pitch_torque_scale)
        self.ankle_roll_torque_scale = float(ankle_roll_torque_scale)

        if protected_joint_deviation_deg is None:
            protected_joint_deviation_deg = {
                'NECK_Y': 18.0,
                'NECK_P': 16.0,
                'R_SHOULDER_P': 50.0,
                'R_SHOULDER_R': 40.0,
                'R_SHOULDER_Y': 45.0,
                'R_ELBOW_P': 65.0,
                'L_SHOULDER_P': 50.0,
                'L_SHOULDER_R': 40.0,
                'L_SHOULDER_Y': 45.0,
                'L_ELBOW_P': 65.0
            }

        self.protected_joint_constraints = []
        if enable_protected_joint_constraints:
            q_lower = self.robot.getPositionLowerLimits()
            q_upper = self.robot.getPositionUpperLimits()
            q_nominal = self.robot.getPositions()

            for joint_name, max_dev_deg in protected_joint_deviation_deg.items():
                try:
                    dof_idx = self.robot.getDof(joint_name).getIndexInSkeleton()
                except Exception:
                    continue

                dev = np.deg2rad(max_dev_deg)
                q_min = max(q_lower[dof_idx], q_nominal[dof_idx] - dev)
                q_max = min(q_upper[dof_idx], q_nominal[dof_idx] + dev)
                if q_min < q_max:
                    self.protected_joint_constraints.append((dof_idx, q_min, q_max, joint_name))

        self.q_lower_limits = np.asarray(self.robot.getPositionLowerLimits(), dtype=float)
        self.knee_dof_indices = {}
        for knee_name in ['L_KNEE_P', 'R_KNEE_P']:
            try:
                self.knee_dof_indices[knee_name] = self.robot.getDof(knee_name).getIndexInSkeleton()
            except Exception:
                continue

        if len(self.knee_dof_indices) < 2:
            self.enable_knee_safety = False

        self.knee_min_support = {}
        self.knee_min_ds = {}
        self.knee_min_swing = {}
        if self.enable_knee_safety:
            support_target = np.deg2rad(knee_min_support_deg)
            ds_target = np.deg2rad(knee_min_ds_deg)
            for knee_name, dof_idx in self.knee_dof_indices.items():
                knee_lower = float(self.q_lower_limits[dof_idx])
                self.knee_min_support[knee_name] = max(knee_lower + np.deg2rad(1.5), support_target)
                self.knee_min_ds[knee_name] = max(knee_lower + np.deg2rad(1.0), ds_target)
                self.knee_min_swing[knee_name] = knee_lower + np.deg2rad(0.5)

        # define sizes for QP solver
        self.num_contacts = 2
        self.num_contact_dims = self.num_contacts * 6
        self.n_vars = 2 * self.dofs + self.num_contact_dims

        self.n_eq_constraints = self.dofs
        self.n_contact_ineq_constraints = 9 * self.num_contacts
        self.n_knee_ineq_constraints = len(self.knee_dof_indices) if self.enable_knee_safety else 0
        self.n_joint_ineq_constraints = 2 * len(self.protected_joint_constraints) + self.n_knee_ineq_constraints
        self.n_ineq_constraints = self.n_contact_ineq_constraints + self.n_joint_ineq_constraints

        # initialize QP solver
        self.qp_solver = QPSolver(self.n_vars, self.n_eq_constraints, self.n_ineq_constraints)

        # selection matrix for redundant dofs
        self.joint_selection = np.zeros((self.dofs, self.dofs))
        for i in range(self.dofs):
            joint_name = self.robot.getDof(i).getName()
            if joint_name in redundant_dofs:
                self.joint_selection[i, i] = 1

        if self.use_model_torque_limits:
            # Use model-provided torque limits for actuated joints (exclude floating base).
            tau_lower_all = np.asarray(self.robot.getForceLowerLimits(), dtype=float)
            tau_upper_all = np.asarray(self.robot.getForceUpperLimits(), dtype=float)
            self.tau_lower_limits = tau_lower_all[6:].copy()
            self.tau_upper_limits = tau_upper_all[6:].copy()

            invalid = (
                ~np.isfinite(self.tau_lower_limits) |
                ~np.isfinite(self.tau_upper_limits) |
                ((self.tau_upper_limits - self.tau_lower_limits) < 1e-3)
            )
            self.tau_lower_limits[invalid] = -100.0
            self.tau_upper_limits[invalid] = 100.0
        else:
            self.tau_lower_limits = -100.0 * np.ones(self.dofs - 6)
            self.tau_upper_limits = 100.0 * np.ones(self.dofs - 6)

        self._apply_ankle_torque_scaling()

    def _scale_joint_torque_limit(self, joint_name, scale):
        if scale <= 0.0:
            return
        try:
            dof_idx = self.robot.getDof(joint_name).getIndexInSkeleton()
        except Exception:
            return

        act_idx = dof_idx - 6
        if act_idx < 0 or act_idx >= len(self.tau_lower_limits):
            return

        lo = float(self.tau_lower_limits[act_idx])
        hi = float(self.tau_upper_limits[act_idx])
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        half_scaled = max(1e-3, half * scale)

        self.tau_lower_limits[act_idx] = center - half_scaled
        self.tau_upper_limits[act_idx] = center + half_scaled

    def _apply_ankle_torque_scaling(self):
        if abs(self.ankle_pitch_torque_scale - 1.0) > 1e-9:
            for joint_name in ['L_ANKLE_P', 'R_ANKLE_P']:
                self._scale_joint_torque_limit(joint_name, self.ankle_pitch_torque_scale)

        if abs(self.ankle_roll_torque_scale - 1.0) > 1e-9:
            for joint_name in ['L_ANKLE_R', 'R_ANKLE_R']:
                self._scale_joint_torque_limit(joint_name, self.ankle_roll_torque_scale)

    def _is_finite_structure(self, obj):
        if isinstance(obj, dict):
            return all(self._is_finite_structure(v) for v in obj.values())
        arr = np.asarray(obj)
        return np.all(np.isfinite(arr))

    def _knee_min_target(self, knee_name, swing_Foot):
        if not self.enable_knee_safety:
            return None

        if swing_Foot == 'ds':
            return self.knee_min_ds[knee_name]

        support_knee = None
        if swing_Foot == 'lfoot':
            support_knee = 'R_KNEE_P'
        elif swing_Foot == 'rfoot':
            support_knee = 'L_KNEE_P'

        if knee_name == support_knee:
            return self.knee_min_support[knee_name]
        return self.knee_min_swing[knee_name]

    def _build_failure_diag(
        self,
        current,
        swing_Foot,
        recovery_mode,
        H,
        b_eq,
        b_ineq,
        optimal_forces,
        f_c_ref,
        contact_l,
        contact_r
    ):
        try:
            H_reg = H + 1e-8 * np.eye(H.shape[0])
            H_cond = np.linalg.cond(H_reg)
        except Exception:
            H_cond = np.nan

        knee_terms = []
        for knee_name in ['L_KNEE_P', 'R_KNEE_P']:
            if knee_name in self.knee_dof_indices:
                idx = self.knee_dof_indices[knee_name]
                knee_deg = np.rad2deg(current['joint']['pos'][idx])
                knee_terms.append(f"{knee_name}:{knee_deg:.1f}")
        knee_str = ','.join(knee_terms) if knee_terms else 'n/a'

        chest_terms = []
        for chest_name in ['CHEST_Y', 'CHEST_P']:
            try:
                idx = self.robot.getDof(chest_name).getIndexInSkeleton()
                chest_terms.append(f"{chest_name}:{np.rad2deg(current['joint']['pos'][idx]):.1f}")
            except Exception:
                continue
        chest_str = ','.join(chest_terms) if chest_terms else 'n/a'

        fz_l = float(f_c_ref[5]) if len(f_c_ref) >= 6 else np.nan
        fz_r = float(f_c_ref[11]) if len(f_c_ref) >= 12 else np.nan

        return (
            f"diag[profile={self.profile_name},recovery={int(bool(recovery_mode))},"
            f"swing={swing_Foot},contact=({int(contact_l)},{int(contact_r)}),"
            f"knee_deg=({knee_str}),chest_deg=({chest_str}),"
            f"fz_ref=({fz_l:.1f},{fz_r:.1f}),"
            f"|f_mpc|={np.linalg.norm(optimal_forces):.1f},"
            f"cond(H)={H_cond:.2e},|b_eq|={np.linalg.norm(b_eq):.2e},"
            f"b_ineq[min,max]=({np.min(b_ineq):.2e},{np.max(b_ineq):.2e}),"
            f"tau_clip={'model' if self.use_model_torque_limits else 'legacy100'}]"
        )

    def get_joint_torques(self, desired, current, swing_Foot, optimal_forces, recovery_mode=False):
        if not self._is_finite_structure(desired) or not self._is_finite_structure(current):
            return np.zeros(self.dofs - 6), False, "Non-finite desired/current state in WBC"
        if not np.all(np.isfinite(np.asarray(optimal_forces))):
            return np.zeros(self.dofs - 6), False, "Non-finite optimal forces in WBC"

        # 1. Contact phase detection (numeric booleans)
        contact_l = 1.0 if (swing_Foot == 'ds' or swing_Foot == 'rfoot') else 0.0
        contact_r = 1.0 if (swing_Foot == 'ds' or swing_Foot == 'lfoot') else 0.0
        lsole = self.robot.getBodyNode('l_sole')
        rsole = self.robot.getBodyNode('r_sole')
        torso = self.robot.getBodyNode('torso')
        base  = self.robot.getBodyNode('body')
        # 2. MPC force transformation (point forces) -> 6D wrenches for WBC
        # MPC runs in the world frame, so moments are computed in the world frame
        d = self.d 
        f0, f1 = optimal_forces[0:3], optimal_forces[3:6]   
        f2, f3 = optimal_forces[6:9], optimal_forces[9:12] 

        R_l = lsole.getTransform().rotation()
        R_r = rsole.getTransform().rotation()
        R_lsole_6x6 = block_diag(R_l, R_l)
        R_rsole_6x6 = block_diag(R_r, R_r)
        R_torso = torso.getTransform().rotation()
        R_base = base.getTransform().rotation()

        torque_l = np.zeros(3)
        force_l = np.zeros(3)
        torque_r = np.zeros(3)
        force_r = np.zeros(3)

        idx = 0
        for x_sign in [1, -1]:
            for y_sign in [1, -1]:
                f_i = optimal_forces[idx*3 : (idx+1)*3]
                r_i = R_l @ np.array([x_sign * self.d, y_sign * self.d, 0])
                torque_l += np.cross(r_i, f_i)
                force_l += f_i
                idx += 1
                
        for x_sign in [1, -1]:
            for y_sign in [1, -1]:
                f_i = optimal_forces[idx*3 : (idx+1)*3]
                r_i = R_r @ np.array([x_sign * self.d, y_sign * self.d, 0])
                torque_r += np.cross(r_i, f_i)
                force_r += f_i
                idx += 1
                
        wrench_l_world = np.concatenate((torque_l, force_l))
        wrench_r_world = np.concatenate((torque_r, force_r))
        
        # Convert to local frame for QP contact variables (A_foot expects local)
        wrench_l_local = R_lsole_6x6.T @ wrench_l_world
        wrench_r_local = R_rsole_6x6.T @ wrench_r_world
        
        f_c_ref = np.concatenate((wrench_l_local, wrench_r_local))
        
        if swing_Foot == 'lfoot':
            f_c_ref[:6] *= 0.0
        elif swing_Foot == 'rfoot':
            f_c_ref[6:] *= 0.0

        # 4. WEIGHT AND GAIN TUNING (updated for SRBD)
        tasks = ['lfoot', 'rfoot', 'com', 'torso', 'base', 'joints']

        weights = {'lfoot': 5., 'rfoot': 5., 'com': 10., 'torso': 4.0, 'base': 2.0, 'joints': 4.5}
        
        pos_gains = {'lfoot': 500., 'rfoot': 500., 'com': 100., 'torso': 80., 'base': 50., 'joints': 90.0}
        vel_gains = {'lfoot': 60., 'rfoot': 60., 'com': 20., 'torso': 20., 'base': 10., 'joints': 14.0}

        if recovery_mode:
            # Reduce upper-body aggressiveness during the post-impact transient.
            weights['torso'] = 1.8
            weights['base'] = 0.8
            weights['joints'] = 2.8

            pos_gains['torso'] = 45.0
            pos_gains['base'] = 20.0
            pos_gains['joints'] = 70.0

            vel_gains['torso'] = 12.0
            vel_gains['base'] = 6.0
            vel_gains['joints'] = 10.0
        
        W_force_track = 1.0

        # 5. Jacobians and derivatives (rotated in the world frame for tasks)
        J_task = {
            'lfoot' : R_lsole_6x6 @ self.robot.getJacobian(lsole),
            'rfoot' : R_rsole_6x6 @ self.robot.getJacobian(rsole),
            'com'   : self.robot.getCOMLinearJacobian(),
            'torso' : R_torso @ self.robot.getAngularJacobian(torso),
            'base'  : R_base @ self.robot.getAngularJacobian(base),
            'joints': self.joint_selection
        }

        Jdot_task = {
            'lfoot' : R_lsole_6x6 @ self.robot.getJacobianClassicDeriv(lsole),
            'rfoot' : R_rsole_6x6 @ self.robot.getJacobianClassicDeriv(rsole),
            'com'   : self.robot.getCOMLinearJacobianDeriv(),
            'torso' : R_torso @ self.robot.getAngularJacobianDeriv(torso),
            'base'  : R_base @ self.robot.getAngularJacobianDeriv(base),
            'joints': np.zeros((self.dofs, self.dofs))
        }

        # Contact Jacobians (in body frame for A_eq and A_foot)
        Jc_lfoot = self.robot.getJacobian(lsole)
        Jc_rfoot = self.robot.getJacobian(rsole)

        # 6. Feedforward, position errors, and velocity errors
        ff = {
            'lfoot' : desired['lfoot']['acc'].flatten(),
            'rfoot' : desired['rfoot']['acc'].flatten(),
            'com'   : desired['com']['acc'].flatten(),
            'torso' : desired['torso']['acc'].flatten(),
            'base'  : desired['base']['acc'].flatten(),
            'joints': desired['joint']['acc'].flatten()
        }

        pos_error = {
            'lfoot' : pose_difference(desired['lfoot']['pos'], current['lfoot']['pos']).flatten(), # 6D
            'rfoot' : pose_difference(desired['rfoot']['pos'], current['rfoot']['pos']).flatten(), # 6D
            'com'   : (desired['com']['pos'] - current['com']['pos']).flatten()[:3],
            'joints': (desired['joint']['pos'] - current['joint']['pos']).flatten()
        }

        # Specific correction for torso and base (from 3x3 matrix to 3D error)
        # Use the rotation logarithm of R_des * R_curr^T to obtain the rotation axis
        pos_error['torso'] = rotation_error(desired['torso']['pos'], current['torso']['pos']).flatten()
        pos_error['base']  = rotation_error(desired['base']['pos'], current['base']['pos']).flatten()

        vel_error = {
            'lfoot' : (desired['lfoot']['vel'] - current['lfoot']['vel']).flatten(),
            'rfoot' : (desired['rfoot']['vel'] - current['rfoot']['vel']).flatten(),
            'com'   : (desired['com']['vel'] - current['com']['vel']).flatten(),
            'torso' : (desired['torso']['vel'] - current['torso']['vel']).flatten(),
            'base'  : (desired['base']['vel'] - current['base']['vel']).flatten(),
            'joints': (desired['joint']['vel'] - current['joint']['vel']).flatten()
        }

        # 7. QP cost function construction
        H = np.zeros((self.n_vars, self.n_vars))
        F = np.zeros(self.n_vars)
        q_ddot_indices = np.arange(self.dofs)
        tau_indices = np.arange(self.dofs, 2 * self.dofs)
        f_c_indices = np.arange(2 * self.dofs, self.n_vars)

        for task in tasks:
            # Task objective: J*q_ddot + Jdot*q_dot = acc_des
            H_task = weights[task] * J_task[task].T @ J_task[task]
            acc_des = ff[task] + vel_gains[task] * vel_error[task] + pos_gains[task] * pos_error[task]
            F_task = - weights[task] * J_task[task].T @ (acc_des - Jdot_task[task] @ current['joint']['vel'])

            H[np.ix_(q_ddot_indices, q_ddot_indices)] += H_task
            F[q_ddot_indices] += F_task

        # Enforce MPC force tracking and regularize tau
        H[np.ix_(f_c_indices, f_c_indices)] += np.eye(len(f_c_indices)) * W_force_track
        F[f_c_indices] += - W_force_track * f_c_ref
        
        W_tau = 1e-4 # Penalty on tau to avoid excessive actuation
        H[np.ix_(tau_indices, tau_indices)] += np.eye(self.dofs) * W_tau

        # 8. Dynamics constraints: M * q_ddot + C + G = tau + Jc^T * fc
        inertia_matrix = self.robot.getMassMatrix()
        actuation_matrix = block_diag(np.zeros((6, 6)), np.eye(self.dofs - 6))
        
        # Contact Jacobian in the body frame
        Jc = np.vstack((contact_l * Jc_lfoot, contact_r * Jc_rfoot))
        
        A_eq = np.hstack((inertia_matrix, - actuation_matrix, - Jc.T))
        b_eq = - self.robot.getCoriolisAndGravityForces()

        # 9. Inequality constraints (friction cone and COP)
        A_ineq = np.zeros((self.n_ineq_constraints, self.n_vars))
        b_ineq = np.zeros(self.n_ineq_constraints)
        
        # Matrix for friction-cone and stability constraints for one foot
        A_foot = np.array([
            [ 1, 0, 0, 0, 0, -self.d], [ -1, 0, 0, 0, 0, -self.d], # COP X
            [ 0, 1, 0, 0, 0, -self.d], [  0, -1, 0, 0, 0, -self.d], # COP Y
            [ 0, 0, 0, 1, 0, -self.µ], [  0, 0, 0, -1, 0, -self.µ], # Friction X
            [ 0, 0, 0, 0, 1, -self.µ], [  0, 0, 0, 0, -1, -self.µ], # Friction Y
            [ 0, 0, 0, 0, 0, -1.0]                                  # f_z >= 0 -> -f_z <= 0
        ])
        A_ineq[:self.n_contact_ineq_constraints, f_c_indices] = block_diag(A_foot, A_foot)

        # 9b. Hard posture constraints: limits on q_{k+1} for neck and arms
        # q_{k+1} = q + dt*q_dot + 0.5*dt^2*q_ddot
        if self.n_joint_ineq_constraints > 0:
            dt = self.control_dt
            start_row = self.n_contact_ineq_constraints
            for j, (dof_idx, q_min, q_max, _) in enumerate(self.protected_joint_constraints):
                q_now = current['joint']['pos'][dof_idx]
                qd_now = current['joint']['vel'][dof_idx]

                qdd_max = 2.0 * (q_max - q_now - dt * qd_now) / (dt * dt)
                qdd_min = 2.0 * (q_min - q_now - dt * qd_now) / (dt * dt)

                row_up = start_row + 2 * j
                row_low = row_up + 1

                A_ineq[row_up, dof_idx] = 1.0
                b_ineq[row_up] = qdd_max

                A_ineq[row_low, dof_idx] = -1.0
                b_ineq[row_low] = -qdd_min

            # 9c. Knee anti-hyperextension safety for the support leg(s).
            if self.n_knee_ineq_constraints > 0:
                knee_row_start = start_row + 2 * len(self.protected_joint_constraints)
                for j, knee_name in enumerate(['L_KNEE_P', 'R_KNEE_P']):
                    if knee_name not in self.knee_dof_indices:
                        continue
                    dof_idx = self.knee_dof_indices[knee_name]
                    q_now = current['joint']['pos'][dof_idx]
                    qd_now = current['joint']['vel'][dof_idx]
                    q_min_target = self._knee_min_target(knee_name, swing_Foot)

                    qdd_min = 2.0 * (q_min_target - q_now - dt * qd_now) / (dt * dt)
                    row = knee_row_start + j
                    A_ineq[row, dof_idx] = -1.0
                    b_ineq[row] = -qdd_min

        # 10. QP solve and torque saturation
        if not (
            np.all(np.isfinite(H)) and
            np.all(np.isfinite(F)) and
            np.all(np.isfinite(A_eq)) and
            np.all(np.isfinite(b_eq)) and
            np.all(np.isfinite(A_ineq)) and
            np.all(np.isfinite(b_ineq))
        ):
            diag = self._build_failure_diag(
                current,
                swing_Foot,
                recovery_mode,
                H,
                b_eq,
                b_ineq,
                optimal_forces,
                f_c_ref,
                contact_l,
                contact_r
            )
            return np.zeros(self.dofs - 6), False, f"Non-finite QP matrices in WBC | {diag}"

        self.qp_solver.set_values(H, F, A_eq, b_eq, A_ineq, b_ineq)
        solution = self.qp_solver.solve()
        if solution is None or not np.all(np.isfinite(solution)):
            print("WBC QP Solver fallito! Restituisco coppie nulle per sicurezza.")
            diag = self._build_failure_diag(
                current,
                swing_Foot,
                recovery_mode,
                H,
                b_eq,
                b_ineq,
                optimal_forces,
                f_c_ref,
                contact_l,
                contact_r
            )
            return np.zeros(self.dofs - 6), False, f"WBC QP solver failed | {diag}"

        if self.debug_verbose:
            print("solution fc:", solution[f_c_indices])
            print("solution tau max:", np.max(np.abs(solution[tau_indices])))
            print("==============================")

        tau = solution[tau_indices]
        joint_torques = tau[6:] 

        if self.enable_knee_safety:
            for knee_name in ['L_KNEE_P', 'R_KNEE_P']:
                if knee_name not in self.knee_dof_indices:
                    continue

                dof_idx = self.knee_dof_indices[knee_name]
                joint_idx = dof_idx - 6
                if joint_idx < 0 or joint_idx >= len(joint_torques):
                    continue

                q_now = current['joint']['pos'][dof_idx]
                qd_now = current['joint']['vel'][dof_idx]
                q_pred = q_now + self.control_dt * qd_now

                q_min_target = self._knee_min_target(knee_name, swing_Foot)
                q_guard = q_min_target + self.knee_output_guard_margin

                if q_pred < q_guard or qd_now < -0.05:
                    err = max(0.0, q_guard - q_pred)
                    corrective = self.knee_output_guard_kp * err + self.knee_output_guard_kd * max(0.0, -qd_now)
                    corrective = min(corrective, self.knee_output_guard_max_torque)
                    # Safety net: do not allow extension torques to dominate when knee is near hyperextension.
                    joint_torques[joint_idx] = max(joint_torques[joint_idx], corrective)
        
       
        return np.clip(joint_torques, self.tau_lower_limits, self.tau_upper_limits), True, ""