import numpy as np
import dartpy as dart
import copy
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
            "NECK_Y", "NECK_P", \
            "R_SHOULDER_P", "R_SHOULDER_R", "R_SHOULDER_Y", "R_ELBOW_P", \
            "L_SHOULDER_P", "L_SHOULDER_R", "L_SHOULDER_Y", "L_ELBOW_P"]
        
        # initialize inverse dynamics
        self.id = id.InverseDynamics(self.hrp4, redundant_dofs)

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

        # initialize logger and plots
        self.logger = Logger(self.initial)
        self.logger.initialize_plot(frequency=10)
        
    def customPreStep(self):
        self.current = self.retrieve_state() # Get current state from the simulation
        planner_tick = int(round(self.time / self.params['world_time_step']))
        # --- AGGIORNAMENTO DINAMICO INERZIA PER SRBD-MPC ---
        self.current['inertia'] = self.base.getInertia().getMoment()
        print("inertia used by MPC:")
        print(self.current['inertia'])
        print("--------------------")
        # --------------------------------------------------

        # 1. Calling control computation (MPC)
        optimal_forces, target_state, _ = self.mpc.compute_controls(self.current, self.time)
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

        # Solver still expects ZMP references
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
        commands = self.id.get_joint_torques(self.desired, self.current, swing_foot_id, optimal_forces)
        
        # --- DEBUG BLOCK: WBC TORQUES ---
        max_tau = np.max(np.abs(commands))
        print(f"Time: {self.time} | Phase: {phase_now} | Swing Foot: {swing_foot_id}")
        print(f"Coppia max: {max_tau:.1f} Nm")
        print("-" * 20)
        
        # 7. Apply torques!
        for i in range(self.params['dof'] - 6):
            self.hrp4.setCommand(i + 6, commands[i])

        # 8. Logger update
        current_for_log = copy.deepcopy(self.current)
        if 'inertia' in current_for_log:
            del current_for_log['inertia'] 
        self.logger.log_data(current_for_log, self.desired)
    
        self.time += self.params['world_time_step']
     

    def retrieve_state(self):
        # 1. Posizioni e orientamenti (Matrici 3x3 per Torso e Base)
        com_position = self.hrp4.getCOM()
        
        # Recuperiamo le matrici di rotazione invece dei vettori
        torso_rot_matrix = self.hrp4.getBodyNode('torso').getTransform(
            withRespectTo=dart.dynamics.Frame.World(), 
            inCoordinatesOf=dart.dynamics.Frame.World()).rotation()
        
        base_rot_matrix = self.base.getTransform(
            withRespectTo=dart.dynamics.Frame.World(), 
            inCoordinatesOf=dart.dynamics.Frame.World()).rotation()

        # 2. Pose dei piedi (Manteniamo rotvec + pos per compatibilità con FTG)
        l_foot_transform = self.lsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        l_foot_orientation = get_rotvec(l_foot_transform.rotation())
        l_foot_position = l_foot_transform.translation()
        left_foot_pose = np.hstack((l_foot_orientation, l_foot_position))
        
        r_foot_transform = self.rsole.getTransform(withRespectTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_orientation = get_rotvec(r_foot_transform.rotation())
        r_foot_position = r_foot_transform.translation()
        right_foot_pose = np.hstack((r_foot_orientation, r_foot_position))

        # 3. Velocità
        com_velocity = self.hrp4.getCOMLinearVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        torso_angular_velocity = self.hrp4.getBodyNode('torso').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        base_angular_velocity = self.hrp4.getBodyNode('body').getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        
        l_foot_spatial_velocity = self.lsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())
        r_foot_spatial_velocity = self.rsole.getSpatialVelocity(relativeTo=dart.dynamics.Frame.World(), inCoordinatesOf=dart.dynamics.Frame.World())

        # 4. Calcolo forze di contatto e ZMP
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
            
            # Clipping per stabilità numerica
            midpoint = (l_foot_position + r_foot_position) / 2.
            zmp = np.clip(zmp, midpoint - 0.3, midpoint + 0.3)
        
        # 5. Dati specifici per SRBD-MPC (Quaternioni e Omega)
        quat_xyzw = R.from_matrix(base_rot_matrix).as_quat()
        quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        # In retrieve_state
        omega = self.base.getAngularVelocity(relativeTo=dart.dynamics.Frame.World(), 
                                     inCoordinatesOf=dart.dynamics.Frame.World())
        
        # 6. Creazione del dizionario di stato
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
            'torso': {'pos': torso_rot_matrix, # Cambiato in Matrice 3x3
                      'vel': torso_angular_velocity,
                      'acc': np.zeros(3)},
            'base' : {'pos': base_rot_matrix,  # Cambiato in Matrice 3x3
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
    #cambio time stamp da 0.01 -0-001
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
    node.setTargetRealTimeFactor(10) # speed up the visualization by 10x
    viewer.addWorldNode(node)

    #viewer.setUpViewInWindow(0, 0, 1920, 1080)
    viewer.setUpViewInWindow(0, 0, 1280, 720)
    #viewer.setUpViewInWindow(0, 0, 640, 480)
    viewer.setCameraHomePosition([5., -1., 1.5],
                                 [1.,  0., 0.5],
                                 [0.,  0., 1. ])
    viewer.run()
