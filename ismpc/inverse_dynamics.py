import dartpy as dart
import numpy as np
from utils import *

class InverseDynamics:
    def __init__(self, robot, redundant_dofs, foot_size=0.1, µ=0.5):
        self.robot = robot
        self.dofs = self.robot.getNumDofs()
        self.d = foot_size / 2.
        self.µ = µ

        # define sizes for QP solver
        self.num_contacts = 2
        self.num_contact_dims = self.num_contacts * 6
        self.n_vars = 2 * self.dofs + self.num_contact_dims

        self.n_eq_constraints = self.dofs
        self.n_ineq_constraints = 9 * self.num_contacts

        # initialize QP solver
        self.qp_solver = QPSolver(self.n_vars, self.n_eq_constraints, self.n_ineq_constraints)

        # selection matrix for redundant dofs
        self.joint_selection = np.zeros((self.dofs, self.dofs))
        for i in range(self.dofs):
            joint_name = self.robot.getDof(i).getName()
            if joint_name in redundant_dofs:
                self.joint_selection[i, i] = 1

    def get_joint_torques(self, desired, current, contact, optimal_forces):
        # 1. Identificazione fasi di contatto (booleani numerici)
        contact_l = 1.0 if (contact == 'ds' or contact == 'rfoot') else 0.0
        contact_r = 1.0 if (contact == 'ds' or contact == 'lfoot') else 0.0
        lsole = self.robot.getBodyNode('l_sole')
        rsole = self.robot.getBodyNode('r_sole')
        torso = self.robot.getBodyNode('torso')
        base  = self.robot.getBodyNode('body')
        # 2. Trasformazione Forze MPC (Puntuali) -> Wrench (6D) per la WBC
        # L'MPC lavora nel World Frame, quindi calcoliamo i momenti nel World Frame
        d = self.d 
        f0, f1 = optimal_forces[0:3], optimal_forces[3:6]   # Punti contatto piede Sx
        f2, f3 = optimal_forces[6:9], optimal_forces[9:12] # Punti contatto piede Dx

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
        
        if contact == 'lfoot':
            f_c_ref[:6] *= 0.0
        elif contact == 'rfoot':
            f_c_ref[6:] *= 0.0

        # 4. TUNING PESI E GUADAGNI (Aggiornati per SRBD)
        tasks = ['lfoot', 'rfoot', 'com', 'torso', 'base', 'joints']

        weights = {'lfoot': 5., 'rfoot': 5., 'com': 10., 'torso': 2.0, 'base': 2.0, 'joints': 1.0}

        pos_gains = {'lfoot': 500., 'rfoot': 500., 'com': 100., 'torso': 50., 'base': 50., 'joints': 50.0}
        vel_gains = {'lfoot': 60., 'rfoot': 60., 'com': 20., 'torso': 10., 'base': 10., 'joints': 5.0}
        
        W_force_track = 1.0

        # 5. Jacobiani e Derivate (Ruotati nel World Frame per i Task)
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

        # Jacobiani di Contatto (In Body Frame per A_eq e A_foot)
        Jc_lfoot = self.robot.getJacobian(lsole)
        Jc_rfoot = self.robot.getJacobian(rsole)

        # 6. Feedforward, Errori di Posizione e Velocità
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

        # Correzione specifica per Torso e Base (da Matrice 3x3 a Errore 3D)
        # Usiamo la funzione logaritmica della rotazione R_des * R_curr^T per ottenere l'asse di rotazione
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

        # 7. Costruzione della Funzione Costo QP
        H = np.zeros((self.n_vars, self.n_vars))
        F = np.zeros(self.n_vars)
        q_ddot_indices = np.arange(self.dofs)
        tau_indices = np.arange(self.dofs, 2 * self.dofs)
        f_c_indices = np.arange(2 * self.dofs, self.n_vars)

        for task in tasks:
            # Task obiettivo: J*q_ddot + Jdot*q_dot = acc_des
            H_task = weights[task] * J_task[task].T @ J_task[task]
            acc_des = ff[task] + vel_gains[task] * vel_error[task] + pos_gains[task] * pos_error[task]
            F_task = - weights[task] * J_task[task].T @ (acc_des - Jdot_task[task] @ current['joint']['vel'])

            H[np.ix_(q_ddot_indices, q_ddot_indices)] += H_task
            F[q_ddot_indices] += F_task

        # Forza l'inseguimento delle forze MPC e regolarizza Tau
        H[np.ix_(f_c_indices, f_c_indices)] += np.eye(len(f_c_indices)) * W_force_track
        F[f_c_indices] += - W_force_track * f_c_ref
        
        W_tau = 1e-4 # Penalità su Tau per non farlo sbracciare!
        H[np.ix_(tau_indices, tau_indices)] += np.eye(self.dofs) * W_tau

        # 8. Vincoli di Dinamica: M * q_ddot + C + G = tau + Jc^T * fc
        inertia_matrix = self.robot.getMassMatrix()
        actuation_matrix = block_diag(np.zeros((6, 6)), np.eye(self.dofs - 6))
        
        # Jacobiano di contatto nel BODY FRAME
        Jc = np.vstack((contact_l * Jc_lfoot, contact_r * Jc_rfoot))
        
        A_eq = np.hstack((inertia_matrix, - actuation_matrix, - Jc.T))
        b_eq = - self.robot.getCoriolisAndGravityForces()

        # 9. Vincoli di Ineguaglianza (Cono attrito e COP)
        A_ineq = np.zeros((self.n_ineq_constraints, self.n_vars))
        b_ineq = np.zeros(self.n_ineq_constraints)
        
        # Matrice per il cono di attrito e stabilità per un singolo piede
        A_foot = np.array([
            [ 1, 0, 0, 0, 0, -self.d], [ -1, 0, 0, 0, 0, -self.d], # COP X
            [ 0, 1, 0, 0, 0, -self.d], [  0, -1, 0, 0, 0, -self.d], # COP Y
            [ 0, 0, 0, 1, 0, -self.µ], [  0, 0, 0, -1, 0, -self.µ], # Attrito X
            [ 0, 0, 0, 0, 1, -self.µ], [  0, 0, 0, 0, -1, -self.µ], # Attrito Y
            [ 0, 0, 0, 0, 0, -1.0]                                  # f_z >= 0 -> -f_z <= 0
        ])
        A_ineq[:, f_c_indices] = block_diag(A_foot, A_foot)

        # 10. Risoluzione QP e Saturazione
        self.qp_solver.set_values(H, F, A_eq, b_eq, A_ineq, b_ineq)
        solution = self.qp_solver.solve()
        print("solution fc:", solution[f_c_indices])
        print("solution tau max:", np.max(np.abs(solution[tau_indices])))
        print("==============================")
        if solution is None:
            print("WBC QP Solver fallito! Restituisco coppie nulle per sicurezza.")
            return np.zeros(self.dofs - 6)

        tau = solution[tau_indices]
        joint_torques = tau[6:] # Rimuovi i 6 DOF della base fluttuante
        
        # Clip finale per proteggere i motori della simulazione
        return np.clip(joint_torques, -100.0, 100.0)