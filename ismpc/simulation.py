import numpy as np
import dartpy as dart
import copy
from datetime import datetime
from utils import *
import os
import srbd_mpc
import footstep_planner
import inverse_dynamics as id
import foot_trajectory_generator as ftg
from logger import Logger
from scipy.spatial.transform import Rotation as R

class Hrp4Controller(dart.gui.osg.RealTimeWorldNode):
    def __init__(self, world, hrp4):
        super(Hrp4Controller, self).__init__(world)
        self.world = world
        self.hrp4 = hrp4
        self.time = 0
        self.debug_verbose = os.environ.get('SIM_DEBUG_VERBOSE', '0').strip().lower() in ['1', 'true', 'yes']
        self.params = {
            'g': 9.81,
            'foot_size': 0.1,
            'step_height': 0.05,
            'ss_duration': 70,
            'ds_duration': 30,
            'world_time_step': world.getTimeStep(),
            'first_swing': 'rfoot',
            'µ': 0.5,
            'N': 50, # Reduced horizon for testing speed maybe, original 100
            'dof': self.hrp4.getNumDofs(),
            'mass': self.hrp4.getMass()
        }
        self.params['initial_stationary_steps'] = int(os.environ.get('INITIAL_STATIONARY_STEPS', '2'))
        self.params['first_step_ds_multiplier'] = int(os.environ.get('FIRST_STEP_DS_MULTIPLIER', '2'))
        self.params['stationary_step_ds_duration'] = int(
            os.environ.get('STATIONARY_STEP_DS_DURATION', str(self.params['ss_duration'] + self.params['ds_duration']))
        )

        # robot links
        self.lsole = hrp4.getBodyNode('l_sole')
        self.rsole = hrp4.getBodyNode('r_sole')
        self.torso = hrp4.getBodyNode('torso')
        self.base  = hrp4.getBodyNode('body')

        self.params['inertia'] = self.base.getInertia().getMoment()

        for i in range(hrp4.getNumJoints()):
            joint = hrp4.getJoint(i)
            dim = joint.getNumDofs()

            # set floating base to passive, everything else to torque
            if   dim == 6: joint.setActuatorType(dart.dynamics.ActuatorType.PASSIVE)
            elif dim == 1: joint.setActuatorType(dart.dynamics.ActuatorType.FORCE)

        # set initial configuration
        initial_configuration = {'CHEST_P': 0., 'CHEST_Y': 0., 'NECK_P': 0., 'NECK_Y': 0., \
                                 'R_HIP_Y': 0., 'R_HIP_R': -3., 'R_HIP_P': -25., 'R_KNEE_P': 50., 'R_ANKLE_P': -25., 'R_ANKLE_R':  3., \
                                 'L_HIP_Y': 0., 'L_HIP_R':  3., 'L_HIP_P': -25., 'L_KNEE_P': 50., 'L_ANKLE_P': -25., 'L_ANKLE_R': -3., \
                                 'R_SHOULDER_P': 4., 'R_SHOULDER_R': -8., 'R_SHOULDER_Y': 0., 'R_ELBOW_P': -25., \
                                 'L_SHOULDER_P': 4., 'L_SHOULDER_R':  8., 'L_SHOULDER_Y': 0., 'L_ELBOW_P': -25.}

        for joint_name, value in initial_configuration.items():
            self.hrp4.setPosition(self.hrp4.getDof(joint_name).getIndexInSkeleton(), value * np.pi / 180.)

        # Bound torso-pelvis relative rotation to keep impact distribution natural.
        chest_limits_deg = {'CHEST_Y': 35.0, 'CHEST_P': 35.0} # Rotation limits YAW and PITCH
        q_lower = self.hrp4.getPositionLowerLimits()
        q_upper = self.hrp4.getPositionUpperLimits()
        q_nominal = self.hrp4.getPositions()
        for joint_name, max_dev_deg in chest_limits_deg.items():
            dof_idx = self.hrp4.getDof(joint_name).getIndexInSkeleton()
            dev = np.deg2rad(max_dev_deg)
            q_lower[dof_idx] = max(q_lower[dof_idx], q_nominal[dof_idx] - dev)
            q_upper[dof_idx] = min(q_upper[dof_idx], q_nominal[dof_idx] + dev)
        self.hrp4.setPositionLowerLimits(q_lower)
        self.hrp4.setPositionUpperLimits(q_upper)

        # position the robot on the ground
        lsole_pos = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        rsole_pos = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World()).translation()
        self.hrp4.setPosition(3, - (lsole_pos[0] + rsole_pos[0]) / 2.)
        self.hrp4.setPosition(4, - (lsole_pos[1] + rsole_pos[1]) / 2.)
        self.hrp4.setPosition(5, - (lsole_pos[2] + rsole_pos[2]) / 2.)

        # initialize state
        self.initial = self.retrieve_state()
        self.contact = 'lfoot' if self.params['first_swing'] == 'rfoot' else 'rfoot' # there is a dummy footstep
        self.desired = copy.deepcopy(self.initial)

        # selection matrix for redundant dofs
        redundant_dofs = [ \
            "CHEST_Y", "CHEST_P", \
            "NECK_Y", "NECK_P", \
            "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P", \
            "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P"]
        
        # initialize inverse dynamics (strict + relaxed fallback)
        # When the "strict" profile is enabled, hard constraints on robot joints angles in the chest,
        # neck and shoulder areas are enforced to prevent unnatural configurations and encourage more human-like recovery strategies.
        # When not necessary, these constraints are lifted to allow the controller more freedom of operation.
        protected_joint_cfg = {
            'NECK_Y': 16.0,
            'NECK_P': 14.0
        }

        use_model_torque_limits = os.environ.get('USE_MODEL_TORQUE_LIMITS', '1').strip().lower() not in ['0', 'false', 'no']
        ankle_pitch_torque_scale = float(os.environ.get('ANKLE_P_TORQUE_SCALE', '1.0'))
        ankle_roll_torque_scale = float(os.environ.get('ANKLE_R_TORQUE_SCALE', '1.0'))
        print(f"[CONFIG] use_model_torque_limits={use_model_torque_limits}")
        print(f"[CONFIG] ankle_torque_scale pitch={ankle_pitch_torque_scale:.2f} roll={ankle_roll_torque_scale:.2f}")

        self.id_strict = id.InverseDynamics(
            self.hrp4,
            redundant_dofs,
            control_dt=self.params['world_time_step'],
            protected_joint_deviation_deg=protected_joint_cfg,
            enable_protected_joint_constraints=True,
            use_model_torque_limits=use_model_torque_limits,
            knee_min_support_deg=18.0,
            knee_min_ds_deg=12.0,
            ankle_pitch_torque_scale=ankle_pitch_torque_scale,
            ankle_roll_torque_scale=ankle_roll_torque_scale,
            profile_name='strict'
        )
        self.id_relaxed = id.InverseDynamics(
            self.hrp4,
            redundant_dofs,
            control_dt=self.params['world_time_step'],
            enable_protected_joint_constraints=False,
            use_model_torque_limits=use_model_torque_limits,
            knee_min_support_deg=24.0,
            knee_min_ds_deg=16.0,
            ankle_pitch_torque_scale=ankle_pitch_torque_scale,
            ankle_roll_torque_scale=ankle_roll_torque_scale,
            profile_name='relaxed'
        )
        
        # Shoulder Indices for Arm Swing
        self.r_shoulder_p_idx = self.hrp4.getDof('R_SHOULDER_P').getIndexInSkeleton()
        self.l_shoulder_p_idx = self.hrp4.getDof('L_SHOULDER_P').getIndexInSkeleton()
        self.chest_y_idx = self.hrp4.getDof('CHEST_Y').getIndexInSkeleton()
        self.chest_p_idx = self.hrp4.getDof('CHEST_P').getIndexInSkeleton()
        self.l_knee_p_idx = self.hrp4.getDof('L_KNEE_P').getIndexInSkeleton()
        self.r_knee_p_idx = self.hrp4.getDof('R_KNEE_P').getIndexInSkeleton()

        # initialize footstep planner
        reference = [(0.1, 0., 0.2)] * 5 + [(0.1, 0., -0.1)] * 10 + [(0.1, 0., 0.)] * 10
        self.footstep_planner = footstep_planner.FootstepPlanner(
            reference,
            self.initial['lfoot']['pos'],
            self.initial['rfoot']['pos'],
            self.params
            )

        # initialize MPC controller (now SRBD!)
        self.mpc = srbd_mpc.SrbdMpc(
            self.initial, 
            self.footstep_planner, 
            self.params
            )

        # initialize foot trajectory generator
        self.foot_trajectory_generator = ftg.FootTrajectoryGenerator(
            self.initial, 
            self.footstep_planner, 
            self.params
            )

        # record initial plan
        self.initial_plan = copy.deepcopy(self.footstep_planner.plan)
        self.nominal_plan = copy.deepcopy(self.footstep_planner.plan)
        self.post_impact_plan = None

        # initialize logger and plots
        self.logger = Logger(self.initial)

        self.shutdown_triggered = False
        self.fall_com_height_threshold = 0.35
        self.max_consecutive_mpc_failures = 8
        self.mpc_fail_count = 0
        disturbance_enabled = os.environ.get('DISTURBANCE_ENABLED', '1').strip().lower() not in ['0', 'false', 'no']
        disturbance_start = float(os.environ.get('DISTURBANCE_START_S', '5.10'))
        disturbance_end = float(os.environ.get('DISTURBANCE_END_S', '5.25'))
        disturbance_magnitude = float(os.environ.get('DISTURBANCE_MAGNITUDE_N', '50.0'))
        disturbance_leftward = os.environ.get('DISTURBANCE_LEFTWARD', '1').strip().lower() not in ['0', 'false', 'no']
        self.disturbance = {
            'enabled': disturbance_enabled,
            'start': disturbance_start,
            'end': disturbance_end,
            'magnitude': disturbance_magnitude,
            'leftward': disturbance_leftward
        }
        self.enable_footstep_replanning = os.environ.get('ENABLE_FOOTSTEP_REPLANNING', '1').strip().lower() not in ['0', 'false', 'no']
        self.replan_only_after_disturbance = os.environ.get('REPLAN_ONLY_AFTER_DISTURBANCE', '1').strip().lower() not in ['0', 'false', 'no']
        self.replan_min_shift_m = float(os.environ.get('REPLAN_MIN_SHIFT_M', '0.005'))
        self.replan_blend_alpha = float(os.environ.get('REPLAN_BLEND_ALPHA', '1.0'))
        self.replan_max_delta_x_m = float(os.environ.get('REPLAN_MAX_DELTA_X_M', '0.25'))
        self.replan_max_delta_y_m = float(os.environ.get('REPLAN_MAX_DELTA_Y_M', '0.25'))
        self.replan_max_step_radius_m = float(os.environ.get('REPLAN_MAX_STEP_RADIUS_M', '0.34'))
        self.replan_min_lateral_clearance_m = float(os.environ.get('REPLAN_MIN_LATERAL_CLEARANCE_M', '0.12'))
        self.replan_max_propagation_shift_m = float(os.environ.get('REPLAN_MAX_PROPAGATION_SHIFT_M', '2.0'))
        self.disturbance_seen = False
        print(f"[CONFIG] disturbance_enabled={self.disturbance['enabled']} start={self.disturbance['start']:.2f}s end={self.disturbance['end']:.2f}s mag={self.disturbance['magnitude']:.1f}N")
        print(f"[CONFIG] footstep_replanning={self.enable_footstep_replanning} replan_only_after_disturbance={self.replan_only_after_disturbance}")
        self.wbc_recovery_duration = 3.0

        self.run_metrics = {
            'max_tau_nm': 0.0,
            'min_com_height_m': float(self.initial['com']['pos'][2]),
            'mpc_fallback_total': 0,
            'wbc_fallback_total': 0,
            'wbc_relaxed_recovery_total': 0,
            'max_chest_y_deg': 0.0,
            'max_chest_p_deg': 0.0,
            'min_l_knee_deg': float(np.rad2deg(self.initial['joint']['pos'][self.l_knee_p_idx])),
            'min_r_knee_deg': float(np.rad2deg(self.initial['joint']['pos'][self.r_knee_p_idx])),
            'com_err_sum_m': 0.0,
            'com_err_samples': 0
        }

        self.save_plots_as_images = True
        self.show_plots_interactive = False
        self.plots_output_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'plots'
        )

    def _print_run_report(self, reason):
        com_samples = self.run_metrics['com_err_samples']
        if com_samples > 0:
            mean_com_err = self.run_metrics['com_err_sum_m'] / com_samples
        else:
            mean_com_err = float('nan')

        print("\n*** RUN SUMMARY ***")
        print(f"Reason: {reason}")
        print(f"Simulated time: {self.time:.3f} s")
        print(f"Max joint torque: {self.run_metrics['max_tau_nm']:.3f} Nm")
        print(f"Min CoM height: {self.run_metrics['min_com_height_m']:.3f} m")
        print(f"MPC fallback count: {self.run_metrics['mpc_fallback_total']}")
        print(f"WBC fallback count: {self.run_metrics['wbc_fallback_total']}")
        print(f"WBC strict->relaxed recoveries: {self.run_metrics['wbc_relaxed_recovery_total']}")
        print(f"Max |CHEST_Y|: {self.run_metrics['max_chest_y_deg']:.2f} deg")
        print(f"Max |CHEST_P|: {self.run_metrics['max_chest_p_deg']:.2f} deg")
        print(f"Min L_KNEE_P: {self.run_metrics['min_l_knee_deg']:.2f} deg")
        print(f"Min R_KNEE_P: {self.run_metrics['min_r_knee_deg']:.2f} deg")
        if np.isfinite(mean_com_err):
            print(f"Mean CoM tracking error: {mean_com_err:.5f} m")
        else:
            print("Mean CoM tracking error: n/a (no valid samples)")

    # Whenever the program reaches termination conditions (either falls, OSQP or IPOPT fails or actually
    # completes the simulation) the program logs all the results and outputs them as graphs in the folder "plots"
    def _shutdown_with_plots(self, reason, exit_code=1):
        if self.shutdown_triggered:
            return
        self.shutdown_triggered = True
        print(f"\n*** TERMINAZIONE ANTICIPATA *** {reason}")
        self._print_run_report(reason)
        if self.post_impact_plan is None:
            self.post_impact_plan = copy.deepcopy(self.footstep_planner.plan)
        self.logger.log_footsteps(self.initial_plan, self.post_impact_plan)

        save_dir = None
        if self.save_plots_as_images:
            run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_dir = os.path.join(self.plots_output_root, f'run_{run_tag}')

        self.logger.show_all_plots(
            save_dir=save_dir,
            show_plots=self.show_plots_interactive
        )
        raise SystemExit(exit_code)

    def _is_finite_state(self, state):
        finite_arrays = [
            state['com']['pos'], state['com']['vel'],
            state['base']['quat'], state['base']['omega'],
            state['joint']['pos'], state['joint']['vel'],
            state['lfoot']['pos'], state['rfoot']['pos']
        ]
        return all(np.all(np.isfinite(np.asarray(a))) for a in finite_arrays)

    def _is_finite_target_state(self, target_state):
        finite_arrays = [
            target_state['com']['pos'], target_state['com']['vel'], target_state['com']['acc'],
            target_state['base']['quat'], target_state['base']['omega']
        ]
        return all(np.all(np.isfinite(np.asarray(a))) for a in finite_arrays)

    def _get_walking_direction_xy(self, planner_tick):
        # Use robot heading (base x-axis projected on ground) so left/right
        # disturbance is truly relative to the robot frame.
        _ = planner_tick
        base_rot = self.current['base']['pos']
        forward = np.array([base_rot[0, 0], base_rot[1, 0]])

        if np.linalg.norm(forward) < 1e-6:
            forward = np.array(self.current['com']['vel'][0:2])
        if np.linalg.norm(forward) < 1e-6:
            forward = np.array([1.0, 0.0])

        return forward / np.linalg.norm(forward)

    def _is_recovery_mode_active(self):
        if not self.disturbance['enabled']:
            return False
        return self.disturbance['start'] <= self.time <= (self.disturbance['end'] + self.wbc_recovery_duration)

    # The robot receives a lateral push when in "ss" phase, 5 seconds in. Total impulse = 7.5 Ns.
    def _apply_lateral_disturbance(self, planner_tick):
        if not self.disturbance['enabled']:
            return
        if not (self.disturbance['start'] < self.time < self.disturbance['end']):
            return

        self.disturbance_seen = True

        forward_xy = self._get_walking_direction_xy(planner_tick)
        lateral_xy = np.array([-forward_xy[1], forward_xy[0]])
        side = 1.0 if self.disturbance['leftward'] else -1.0
        force_xy = side * self.disturbance['magnitude'] * lateral_xy
        ext_force = np.array([force_xy[0], force_xy[1], 0.0])

        self.torso.addExtForce(ext_force, [0.0, 0.0, 0.0], isForceLocal=False, isOffsetLocal=False)
        self.logger.log_disturbance(self.time, ext_force, forward_xy, self.current['com']['pos'][0:2])

        if planner_tick % 5 == 0:
            orthogonality = np.dot(forward_xy, force_xy) / (np.linalg.norm(force_xy) + 1e-9)
            print(
                f" [CRASH TEST] Disturbo laterale applicato | t={self.time:.2f}s "
                f"| F=[{ext_force[0]:.2f}, {ext_force[1]:.2f}, 0.00]N "
                f"| frame=robot-left | dot(forward,force)={orthogonality:.4f}"
            )

    def _should_replan_footstep(self, current_phase, current_step_idx):
        if not self.enable_footstep_replanning:
            return False
        if current_step_idx is None:
            return False
        if current_step_idx + 1 >= len(self.footstep_planner.plan):
            return False
        
        # Abilita il replanning SOLTANTO durante il periodo di instabilità o nel transient dopo la spinta.
        if self.replan_only_after_disturbance and not self._is_recovery_mode_active():
            return False
        return True

    def _sanitize_replanned_swing_target(self, opt_p_swing, current_step_idx):
        if current_step_idx is None or current_step_idx + 1 >= len(self.footstep_planner.plan):
            return np.array(opt_p_swing[:2], dtype=float)

        support_step = self.footstep_planner.plan[current_step_idx]
        support_xy = np.array(support_step['pos'][0:2], dtype=float)
        nominal_xy = np.array(self.nominal_plan[current_step_idx + 1]['pos'][0:2], dtype=float)
        candidate_xy = np.array(opt_p_swing[:2], dtype=float)

        alpha = np.clip(self.replan_blend_alpha, 0.0, 1.0)
        candidate_xy = (1.0 - alpha) * nominal_xy + alpha * candidate_xy

        delta = candidate_xy - nominal_xy
        delta[0] = np.clip(delta[0], -self.replan_max_delta_x_m, self.replan_max_delta_x_m)
        delta[1] = np.clip(delta[1], -self.replan_max_delta_y_m, self.replan_max_delta_y_m)
        candidate_xy = nominal_xy + delta

        step_vec = candidate_xy - support_xy
        step_norm = np.linalg.norm(step_vec)
        if step_norm > self.replan_max_step_radius_m and step_norm > 1e-9:
            candidate_xy = support_xy + (self.replan_max_step_radius_m / step_norm) * step_vec

        # RIMOZIONE LIMITI LATERALI ANTI-COMPENETRAZIONE:
        # Come nell'MPC, permettiamo matematicamente passi incrociati se necessari per recuperare equilibrio.
        # support_id = support_step['foot_id']
        # if support_id == 'lfoot':
        #     candidate_xy[1] = min(candidate_xy[1], support_xy[1] - self.replan_min_lateral_clearance_m)
        # else:
        #     candidate_xy[1] = max(candidate_xy[1], support_xy[1] + self.replan_min_lateral_clearance_m)

        return candidate_xy
        
    def customPreStep(self):
        try:
            self.current = self.retrieve_state() # Get current state from the simulation
        except Exception as exc:
            self._shutdown_with_plots(f"Errore nel retrieve_state: {exc}")
            return

        if not self._is_finite_state(self.current):
            self._shutdown_with_plots("Stato non finito rilevato (NaN/Inf) prima del controllo")
            return

        if self.current['com']['pos'][2] < self.fall_com_height_threshold:
            self._shutdown_with_plots(
                f"Caduta rilevata: altezza CoM={self.current['com']['pos'][2]:.3f} m"
            )
            return

        self.run_metrics['min_com_height_m'] = min(
            self.run_metrics['min_com_height_m'],
            float(self.current['com']['pos'][2])
        )

        chest_y_deg = float(np.rad2deg(self.current['joint']['pos'][self.chest_y_idx]))
        chest_p_deg = float(np.rad2deg(self.current['joint']['pos'][self.chest_p_idx]))
        l_knee_deg = float(np.rad2deg(self.current['joint']['pos'][self.l_knee_p_idx]))
        r_knee_deg = float(np.rad2deg(self.current['joint']['pos'][self.r_knee_p_idx]))
        self.run_metrics['max_chest_y_deg'] = max(self.run_metrics['max_chest_y_deg'], abs(chest_y_deg))
        self.run_metrics['max_chest_p_deg'] = max(self.run_metrics['max_chest_p_deg'], abs(chest_p_deg))
        self.run_metrics['min_l_knee_deg'] = min(self.run_metrics['min_l_knee_deg'], l_knee_deg)
        self.run_metrics['min_r_knee_deg'] = min(self.run_metrics['min_r_knee_deg'], r_knee_deg)

        planner_tick = int(round(self.time / self.params['world_time_step']))

        # Arm swinging logic
        l_foot_x = self.current['lfoot']['pos'][3] # Index 3 is X translation
        r_foot_x = self.current['rfoot']['pos'][3]
        leg_diff_x = l_foot_x - r_foot_x
        
        arm_swing_gain = 1.5 # Gain mappings 
        base_shoulder_pitch = 4.0 * np.pi / 180.0
        
        arm_offset = np.clip(arm_swing_gain * leg_diff_x, -0.45, 0.45)
        self.desired['joint']['pos'][self.r_shoulder_p_idx] = base_shoulder_pitch - arm_offset
        self.desired['joint']['pos'][self.l_shoulder_p_idx] = base_shoulder_pitch + arm_offset

        self.current['inertia'] = self.base.getInertia().getMoment()
        if self.debug_verbose:
            print("inertia used by MPC:")
            print(self.current['inertia'])
            print("--------------------")

        if self.time > 26.0:
            self._shutdown_with_plots("Simulazione completata (26s)", exit_code=0)
            return

        # Record post impact plan roughly 1s after impact
        if self.time > 6.0 and self.post_impact_plan is None:
            self.post_impact_plan = copy.deepcopy(self.footstep_planner.plan)

        # Apply the disturbance (push)
        self._apply_lateral_disturbance(planner_tick)

        # 1. Calling control computation (MPC)
        if not hasattr(self, 'mpc_freq'):
            self.mpc_freq = 5 # 20 Hz. WBC runs at 5 times the speed.
            self.mpc_tick_offset = 0
        
        if self.mpc_tick_offset % self.mpc_freq == 0:
            current_step_idx = self.footstep_planner.get_step_index_at_time(planner_tick)
            current_phase = self.footstep_planner.get_phase_at_time(planner_tick)
            allow_footstep_replanning = self._should_replan_footstep(current_phase, current_step_idx)
            try:
                _, _, _, opt_p_swing = self.mpc.compute_controls(
                    self.current,
                    self.time,
                    nominal_plan=self.nominal_plan,
                    allow_footstep_replanning=allow_footstep_replanning
                )
            except Exception as exc:
                self._shutdown_with_plots(f"Eccezione MPC non gestita: {exc}")
                return

            if self.mpc.last_solve_success:
                self.mpc_fail_count = 0
            else:
                self.mpc_fail_count += 1
                self.run_metrics['mpc_fallback_total'] += 1
                print(f"[MPC] Solve fallito ({self.mpc_fail_count} consecutivi): {self.mpc.last_solve_message}")
                if self.mpc_fail_count >= self.max_consecutive_mpc_failures:
                    self._shutdown_with_plots(
                        f"Arresto preventivo: {self.mpc_fail_count} failure MPC consecutive"
                    )
                    return
            
            # Dynamic recovery: Foot target is updated ONLY DURING FLIGHT (ss)
            if allow_footstep_replanning:
                safe_swing_xy = self._sanitize_replanned_swing_target(opt_p_swing, current_step_idx)
                self.footstep_planner.plan[current_step_idx + 1]['pos'][0] = safe_swing_xy[0]
                self.footstep_planner.plan[current_step_idx + 1]['pos'][1] = safe_swing_xy[1]
                
            self.mpc_tick_offset = 0

        # After landing, whichever shift was chosen for recovery is propagated to the rest of the footstep plan.
                # Riconvergenza alla traiettoria originale mediante decay
        current_step_idx = self.footstep_planner.get_step_index_at_time(planner_tick)
        phase_now = self.footstep_planner.get_phase_at_time(planner_tick)

        if hasattr(self, 'phase_prev') and self.phase_prev == 'ss' and phase_now == 'ds':
            idx_landed = current_step_idx + 1
            if current_step_idx is not None and idx_landed < len(self.initial_plan):
                # Calcoliamo lo shift TOTALE rispetto al piano ORIGINALE (initial_plan)
                final_pos = self.footstep_planner.plan[idx_landed]['pos'][:2]
                orig_pos = self.initial_plan[idx_landed]['pos'][:2]
                total_shift = final_pos - orig_pos

                # Limita lo shift massimo per sicurezza
                shift_norm = np.linalg.norm(total_shift)
                if shift_norm > self.replan_max_propagation_shift_m and shift_norm > 1e-9:
                    total_shift = (self.replan_max_propagation_shift_m / shift_norm) * total_shift
                
                replan_active = (not self.replan_only_after_disturbance) or self.disturbance_seen
                if self.enable_footstep_replanning and replan_active:
                    recovery_rate = 0.7
                    
                    for i in range(idx_landed + 1, len(self.initial_plan)):
                        k = i - idx_landed
                        # Calcoliamo lo shift residuo da applicare alla posizione originale
                        decayed_shift = total_shift * (recovery_rate ** k)
                        
                        # Ripristiniamo la posizione partendo da quella INIZIALE + lo shift residuo
                        self.nominal_plan[i]['pos'][0] = self.initial_plan[i]['pos'][0] + decayed_shift[0]
                        self.nominal_plan[i]['pos'][1] = self.initial_plan[i]['pos'][1] + decayed_shift[1]
                        
                        self.footstep_planner.plan[i]['pos'][0] = self.nominal_plan[i]['pos'][0]
                        self.footstep_planner.plan[i]['pos'][1] = self.nominal_plan[i]['pos'][1]
                        
        self.phase_prev = phase_now


        optimal_forces = self.mpc.get_buffered_forces(self.mpc_tick_offset)
        target_state = self.mpc.get_buffered_state(self.mpc_tick_offset)
        if target_state is None or not self._is_finite_target_state(target_state):
            self._shutdown_with_plots("Target MPC non valido (None/NaN)")
            return

        if not np.all(np.isfinite(np.asarray(optimal_forces))):
            self._shutdown_with_plots("Forze MPC non finite")
            return

        self.mpc_tick_offset += 1
        phase_now = self.footstep_planner.get_phase_at_time(planner_tick)
        step_idx_now = self.footstep_planner.get_step_index_at_time(planner_tick)

        if phase_now == 'ds':
            support_foot_id = 'ds'
            swing_foot_id = 'ds'
        else:
            support_foot_id = self.footstep_planner.plan[step_idx_now]['foot_id']
            swing_foot_id = 'lfoot' if support_foot_id == 'rfoot' else 'rfoot'

        # 3. Update desired state based on MPC output
        self.desired['com']['pos'] = target_state['com']['pos']
        self.desired['com']['vel'] = target_state['com']['vel']
        self.desired['com']['acc'] = target_state['com']['acc']
        com_err = np.linalg.norm(self.desired['com']['pos'] - self.current['com']['pos'])
        if np.isfinite(com_err):
            self.run_metrics['com_err_sum_m'] += float(com_err)
            self.run_metrics['com_err_samples'] += 1

        self.desired['zmp']['pos'] = self.current['zmp']['pos']
        self.desired['zmp']['vel'] = np.zeros(3)

        # 4. Gait generation (FTG)
        feet_trajectories = self.foot_trajectory_generator.generate_feet_trajectories_at_time(planner_tick)
        for foot in ['lfoot', 'rfoot']:
            for key in ['pos', 'vel', 'acc']:
                self.desired[foot][key] = feet_trajectories[foot][key]

        # 5. Body & Orientation reference
        target_quat = target_state['base']['quat']
        target_quat_scipy = [target_quat[1], target_quat[2], target_quat[3], target_quat[0]]
        rot_matrix_target = R.from_quat(target_quat_scipy).as_matrix()

        # Base orientation reference
        self.desired['base']['pos'] = rot_matrix_target
        self.desired['base']['vel'] = target_state['base']['omega']
        self.desired['base']['acc'] = np.zeros(3)
        self.desired['base']['quat'] = target_quat
        self.desired['base']['omega'] = target_state['base']['omega']

        # Torso orientation reference
        self.desired['torso']['pos'] = rot_matrix_target
        self.desired['torso']['vel'] = target_state['base']['omega']
        self.desired['torso']['acc'] = np.zeros(3)

        # 6. WBC computations (Inverse Dynamics)
        # Disabilitiamo il recovery_mode nel WBC per evitare instabilità (snap) sui gain
        # come ipotizzato, evitando che il robot cada "dopo" aver superato la botta.
        wbc_recovery_mode = False
        commands, wbc_ok, wbc_msg = self.id_strict.get_joint_torques(
            self.desired,
            self.current,
            swing_foot_id,
            optimal_forces,
            recovery_mode=wbc_recovery_mode
        )

        if not wbc_ok:
            relaxed_commands, relaxed_ok, relaxed_msg = self.id_relaxed.get_joint_torques(
                self.desired,
                self.current,
                swing_foot_id,
                optimal_forces,
                recovery_mode=wbc_recovery_mode
            )
            if relaxed_ok:
                commands = relaxed_commands
                wbc_ok = True
                self.run_metrics['wbc_relaxed_recovery_total'] += 1
                print(f"[WARN] WBC strict failed, fallback to relaxed succeeded: {wbc_msg}")
            else:
                wbc_msg = f"strict: {wbc_msg} | relaxed: {relaxed_msg}"

        if not wbc_ok:
            self.run_metrics['wbc_fallback_total'] += 1
            self._shutdown_with_plots(f"Arresto preventivo WBC: {wbc_msg}")
            return
        
        # Debug
        max_tau = np.max(np.abs(commands))
        self.run_metrics['max_tau_nm'] = max(self.run_metrics['max_tau_nm'], float(max_tau))
        if self.debug_verbose:
            print(f"Time: {self.time} | Phase: {phase_now} | Swing Foot: {swing_foot_id}")
            print(f"Coppia max: {max_tau:.1f} Nm")
            print("-" * 20)
        
        # 7. Apply torques
        for i in range(self.params['dof'] - 6):
            self.hrp4.setCommand(i + 6, commands[i])

        # 8. Logger update
        current_for_log = copy.deepcopy(self.current)
        if 'inertia' in current_for_log:
            del current_for_log['inertia'] 
        self.logger.log_data(self.desired, current_for_log, optimal_forces, commands)
    
        self.time += self.params['world_time_step']
     

    def retrieve_state(self):
        # 1. Position and orientation of the robot (CoM, torso, base) 
        com_position = self.hrp4.getCOM()
        
        torso_rot_matrix = self.hrp4.getBodyNode('torso').getTransform(
            withRespectTo=dart.dynamics.Frame.World(), 
            inCoordinatesOf=dart.dynamics.Frame.World()).rotation()
        
        base_rot_matrix = self.base.getTransform(
            withRespectTo=dart.dynamics.Frame.World(), 
            inCoordinatesOf=dart.dynamics.Frame.World()).rotation()

        # 2. Feet pose 
        l_foot_transform = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_orientation = get_rotvec(l_foot_transform.rotation())
        l_foot_position = l_foot_transform.translation()
        left_foot_pose = np.hstack((l_foot_orientation, l_foot_position))
        
        r_foot_transform = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_orientation = get_rotvec(r_foot_transform.rotation())
        r_foot_position = r_foot_transform.translation()
        right_foot_pose = np.hstack((r_foot_orientation, r_foot_position))

        # 3. Velocities
        com_velocity = self.hrp4.getCOMLinearVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        torso_angular_velocity = self.hrp4.getBodyNode('torso').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        base_angular_velocity = self.hrp4.getBodyNode('body').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        
        l_foot_spatial_velocity = self.lsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_spatial_velocity = self.rsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())

        # 4. Contact forces, zmp estimation
        force = np.zeros(3)
        collision_result = self.world.getLastCollisionResult()
        for contact in collision_result.getContacts():
            force += contact.force

        zmp = np.zeros(3)
        if force[2] <= 0.1:
            zmp = np.array([0., 0., 0.])
        else:
            zmp[2] = com_position[2] - force[2] / (self.hrp4.getMass() * self.params['g'] / 0.72)
            for contact in collision_result.getContacts():
                if contact.force[2] <= 0.1: continue
                zmp[0] += (contact.point[0] * contact.force[2] / force[2] + (zmp[2] - contact.point[2]) * contact.force[0] / force[2])
                zmp[1] += (contact.point[1] * contact.force[2] / force[2] + (zmp[2] - contact.point[2]) * contact.force[1] / force[2])
            
            # Clipping
            midpoint = (l_foot_position + r_foot_position) / 2.
            zmp = np.clip(zmp, midpoint - 0.3, midpoint + 0.3)
        
        # 5. Quaternions
        quat_xyzw = R.from_matrix(base_rot_matrix).as_quat()
        quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        # In retrieve_state
        omega = self.base.getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), 
                                     inCoordinatesOf=dart.dynamics.Frame.World())
        
        # 6. State dictionary construction
        return {
            'lfoot': {'pos': left_foot_pose,
                      'vel': l_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'rfoot': {'pos': right_foot_pose,
                      'vel': r_foot_spatial_velocity,
                      'acc': np.zeros(6)},
            'com'  : {'pos': com_position,
                      'vel': com_velocity,
                      'acc': np.zeros(3)},
            'torso': {'pos': torso_rot_matrix, 
                      'vel': torso_angular_velocity,
                      'acc': np.zeros(3)},
            'base' : {'pos': base_rot_matrix,  
                      'vel': base_angular_velocity,
                      'acc': np.zeros(3),
                      'quat': quat_wxyz,
                      'omega': omega},
            'joint': {'pos': self.hrp4.getPositions(),
                      'vel': self.hrp4.getVelocities(),
                      'acc': np.zeros(self.params['dof'])},
            'zmp'  : {'pos': zmp,
                      'vel': np.zeros(3),
                      'acc': np.zeros(3)}
        }

if __name__ == "__main__":
    world = dart.simulation.World()

    urdfParser = dart.utils.DartLoader()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hrp4   = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "hrp4.urdf"))
    ground = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "ground.urdf"))
    world.addSkeleton(hrp4)
    world.addSkeleton(ground)
    world.setGravity([0, 0, -9.81])
    world.setTimeStep(0.01)

    # set default inertia
    default_inertia = dart.dynamics.Inertia(1e-8, np.zeros(3), 1e-10 * np.identity(3))
    for body in hrp4.getBodyNodes():
        if body.getMass() == 0.0:
            body.setMass(1e-8)
            body.setInertia(default_inertia)

    node = Hrp4Controller(world, hrp4)

    # create world node and add it to viewer
    viewer = dart.gui.osg.Viewer()
    node.setTargetRealTimeFactor(10) 
    viewer.addWorldNode(node)

    #viewer.setUpViewInWindow(0, 0, 1920, 1080)
    viewer.setUpViewInWindow(0, 0, 1280, 720)
    viewer.setCameraHomePosition([5., -1., 1.5],
                                 [1.,  0., 0.5],
                                 [0.,  0., 1. ])
    viewer.run()
    
    if node.post_impact_plan is None:
        node.post_impact_plan = copy.deepcopy(node.footstep_planner.plan)
    node.logger.log_footsteps(node.initial_plan, node.post_impact_plan)
    node.logger.show_all_plots()
