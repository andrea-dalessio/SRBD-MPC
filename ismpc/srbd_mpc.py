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
        self.footstep_planner = footstep_planner
        self.mass = params['mass']
        self.I = params['inertia']
        self.I_inv = np.linalg.inv(self.I)
        self.mu = params['µ']
        self.g = [0, 0, -params['g']]
               
        # variables for optimization problem
        self.opt = cs.Opti()
        self.X = self.opt.variable(13, self.N + 1) # pos, vel CoM, quaternion, angular vel
        self.U = self.opt.variable(12, self.N) # forces and torques at the feet + flying foot coords
        self.p_swing = self.opt.variable(2) # swing foot position in the world frame
        
        # optimization problem (The quaternion derivative and the total torque functions will be added later)
        self.f = lambda x, u: cs.vertcat(
            x[3:6], # CoM velocity
            (1 / self.mass) * (u[0:3] + u[3:6] + u[6:9] + u[9:12]) + self.g, # CoM acceleration
            self.compute_quaternion_derivative(x[6:10], x[10:13]), # quaternion derivative
            self.I_inv @ (self.compute_total_torque(x, u) - cs.cross(x[10:13], self.I @ x[10:13])) # angular acceleration
        )
        
        p_opts = {"expand": True}
        s_opts = {
            "max_iter": 1000,
            "print_level": 0,
            "sb": "yes",
        }
        self.opt.solver('ipopt', p_opts, s_opts)
        
        
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

    def compute_total_torque(self, x, u):
        

    def apply_kinematic_constraints(self, t):
        current_step_index = self.footstep_planner.get_step_index_at_time(t)
        support_foot_pos = self.footstep_planner.plan[current_step_index]['pos']
        
        L_max = 0.5 # maximum leg length

        self.opt.subject_to( 
            (self.p_swing[0] - support_foot_pos[0])**2 + 
            (self.p_swing[1] - support_foot_pos[1])**2 <= L_max**2 
        )