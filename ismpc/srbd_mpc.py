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
        self.U = self.opt.variable(12, self.N) # forces and torques at the feet
        self.p_swing = self.opt.variable(2) # swing foot position in the world frame
        
        # optimization problem (The quaternion derivative and the total torque functions will be added later)
        self.f = lambda x, u: cs.vertcat(
            x[3:6], # CoM velocity
            (1 / self.mass) * (u[0:3] + u[3:6] + u[6:9] + u[9:12]) + self.g, # CoM acceleration
            self.compute_quaternion_derivative(x[6:10], x[10:13]), # quaternion derivative
            self.I_inv @ (self.compute_total_torque(x, u, p_contacts) - cs.cross(x[10:13], self.I @ x[10:13])) # angular acceleration
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

    def compute_total_torque(self, x, u, p_contacts):
        com_pos = x[0:3]
        total_torque = cs.vertcat(0.0, 0.0, 0.0)
        
        # Iterating on the 4 contact points (2 feet with 2 contact points each)
        for i in range(4):
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
        # Set initial state
        x0 = cs.vertcat(
            current_state['com']['pos'],
            current_state['com']['vel'],
            current_state['base']['quat'],
            current_state['base']['omega']
        )
        self.opt.subject_to(self.X[:, 0] == x0)
        
        current_step_index = self.footstep_planner.get_step_index_at_time(t)
        
        # Stance calculation
        current_support = self.footstep_planner.get_support_foot_at_time(t)
        if current_support == 'lfoot':
            p_stance_xy = current_state['lfoot']['pos'][3:5] 
            p_stance_z  = current_state['lfoot']['pos'][5]
        else:
            p_stance_xy = current_state['rfoot']['pos'][3:5]
            p_stance_z  = current_state['rfoot']['pos'][5]
        
        # Cost setup
        cost = 0.0
        
        W_com_z = 5000.0                      # Weight for tracking the CoM height (important for stability)
        W_vel = np.diag([10.0, 10.0, 10.0])   # Cost for chasing CoM velocity
        W_quat = np.diag([100.0, 100.0, 100.0]) # Cost for maintaining the torso upright (penalizes X,Y,Z of the quaternion)
        W_omega = np.diag([10.0, 10.0, 10.0]) # Cost for stopping rotations
        W_force = np.eye(12) * 1e-4           # Minimum force for the motors
        W_swing = 1000.0                      # Cost for tracking the desired footprint

        h_ref = 0.72 # Reference height for the CoM for the HRP-4        
        
        
        # Main loop
        for k in range(self.N): 
            t_k = t + k * self.delta 
            future_step_index = self.footstep_planner.get_step_index_at_time(t_k)
            phase = self.footstep_planner.get_phase_at_time(t_k)
            support_foot = self.footstep_planner.get_support_foot_at_time(t_k)
            
            # Foot stance
            if future_step_index == current_step_index: 
                # Now standing. Next step -> flight. At every time step feet are immobile
                p_lfoot_k = current_state['lfoot']['pos'][3:5]
                p_rfoot_k = current_state['rfoot']['pos'][3:5]
            else: 
                # Feet landing! It becomes the decision variable
                if current_support == 'lfoot':
                    p_lfoot_k = current_state['lfoot']['pos'][3:5]
                    p_rfoot_k = self.p_swing # Right foot stepped
                else:
                    p_lfoot_k = self.p_swing # Left foot stepped
                    p_rfoot_k = current_state['rfoot']['pos'][3:5]

            p_contacts = self.generate_contact_points(p_lfoot_k, p_rfoot_k, 0.0)

            # Applying Dynamics!
            x_next = self.X[:, k] + self.delta * self.f(self.X[:, k], self.U[:, k], p_contacts)
            self.opt.subject_to(self.X[:, k + 1] == x_next)
            
            # Friction cone enforcement
            for i in range(4):
                fx = self.U[i*3 + 0, k]
                fy = self.U[i*3 + 1, k]
                fz = self.U[i*3 + 2, k]
                
                self.opt.subject_to(fz >= 0) # Forces should point upwards from the ground
                self.opt.subject_to(fx <=  self.mu * fz)
                self.opt.subject_to(fx >= -self.mu * fz)
                self.opt.subject_to(fy <=  self.mu * fz)
                self.opt.subject_to(fy >= -self.mu * fz)
                
            # If flying phase, the FLYING FOOT should have zero forces
            if phase == 'ss':
                if support_foot == 'lfoot':
                    self.opt.subject_to(self.U[6:12, k] == 0)
                else:
                    self.opt.subject_to(self.U[0:6, k] == 0)
            
            # Cost computation section (refer to the paper!)
            # 1. Height tracking for the CoM
            cost += W_com_z * (self.X[2, k + 1] - h_ref)**2
            
            # 2. Regularization terms
            v_err = self.X[3:6, k+1] 
            cost += cs.mtimes([v_err.T, W_vel, v_err])
            
            quat_err = self.X[7:10, k+1]
            cost += cs.mtimes([quat_err.T, W_quat, quat_err])
            
            omega_err = self.X[10:13, k+1]
            cost += cs.mtimes([omega_err.T, W_omega, omega_err])
            
            cost += cs.mtimes([self.U[:, k].T, W_force, self.U[:, k]]) # Minimizing effort
                
        # K.C. for the leg
        self.apply_kinematic_constraints(t)
        
        # Another cost term
        # 3. Contact tracking
        try:
            next_step_target = self.footstep_planner.plan[current_step_index + 1]['pos'][3:5]
        except IndexError:
            next_step_target = self.footstep_planner.plan[current_step_index]['pos'][3:5]   
        cost += W_swing * cs.sumsqr(self.p_swing - next_step_target)
        
        # Solve
        self.opt.minimize(cost)
        sol = self.opt.solve()
        optimal_controls = sol.value(self.U[:, 0]) # Only first step (MPC)
        target_state = self.extract_target_state(sol) # Get where I should be at next step
        contact = self.footstep_planner.get_phase_at_time(t)
        
        return optimal_controls, target_state, contact
    
    def generate_contact_points(self, p_left_xy, p_right_xy, z):
        d = self.foot_size / 2.0
        
        # Left foot
        xl, yl = p_left_xy[0], p_left_xy[1]
        # 1. Back Right
        p0 = cs.vertcat(xl - d, yl - d, z)
        # 2. Front Left
        p1 = cs.vertcat(xl + d, yl + d, z)
        
        # Right foot
        xr, yr = p_right_xy[0], p_right_xy[1]
        # 3. Back Left
        p2 = cs.vertcat(xr - d, yr + d, z)
        # 4. Front Right
        p3 = cs.vertcat(xr + d, yr - d, z)

        return cs.vertcat(p0, p1, p2, p3)
    
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