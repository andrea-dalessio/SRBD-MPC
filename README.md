# SRBD_MPC_Humanoid_Locomotion

## 🤖 **Single Rigid Body MPC for Humanoid Locomotion and Push Recovery**

This project presents the design and validation of a nonlinear control architecture for humanoid locomotion based on **Single Rigid Body Dynamics Model Predictive Control (SRBD-MPC)**.

The robot is modeled through its **Center of Mass (CoM) dynamics** and **torso orientation**, represented using **unit quaternions** to avoid singularities in 3D rotations. Instead of relying on a simplified Linear Inverted Pendulum model, the framework explicitly accounts for the robot’s rotational dynamics, angular velocity, angular momentum variation, and multi-contact Ground Reaction Forces.

The main objective is to generate stable walking motions while improving robustness against lateral disturbances through **reactive footstep replanning**, **multi-contact force optimization**, and a **Whole-Body Control layer based on inverse dynamics**.

---

## 🧭 **Control Philosophy**

The architecture follows a hierarchical control structure:

• A nominal footstep planner provides the reference gait sequence
• The SRBD-MPC optimizes CoM motion, torso attitude, GRFs, and swing foot placement
• A trajectory generator produces smooth swing foot trajectories
• The Whole-Body Controller converts MPC references into joint accelerations and torques

The MPC does not rigidly follow the nominal footstep plan. Instead, the next swing foot target is treated as a soft optimization variable, allowing the robot to adapt its foot placement when external disturbances make the original plan dynamically unsafe.

---

## 🧠 **SRBD-MPC Layer**

The humanoid robot is approximated as a **single rigid body** with state:

```math
x = [p^T,\dot{p}^T,q^T,\omega^T]^T \in \mathbb{R}^{13}
```

where:

• `p` is the CoM position
• `p_dot` is the CoM linear velocity
• `q` is the unit quaternion representing torso orientation
• `omega` is the angular velocity of the base

The control input is composed of the **Ground Reaction Forces (GRFs)** distributed across the contact vertices of the feet:

```math
u \in \mathbb{R}^{24}
```

using **4 contact vertices per foot**, each applying a 3D force.

This multi-contact representation allows the controller to manage:

• Center of Pressure distribution
• Contact torque generation
• Friction constraints
• Unilateral ground contact
• Push recovery through force redistribution

---

## 🦶 **Reactive Footstep Replanning**

During nominal walking, the swing foot follows the landing target provided by the footstep planner.

When a lateral disturbance is detected, the MPC is allowed to modify the next foothold:

```math
\Psi_{swing}=W_{swing}\|p_{swing}-p_{ref}\|^2
```

This term softly anchors the replanned footstep to the nominal one, while still allowing deviations when required for balance recovery.

The replanned step is constrained by the robot’s kinematic workspace:

```math
\|p_{swing}-p_{support}\|^2 \leq L_{max}^2
```

This ensures that the robot can recover from disturbances without requesting physically unreachable leg extensions.

---

## ⚖️ **Cost Function and Stability Objectives**

The MPC minimizes a nonlinear objective function composed of several terms:

• **CoM tracking** to maintain vertical stability and guide horizontal progression
• **Quaternion attitude tracking** to stabilize torso roll, pitch, and yaw
• **Regularization terms** to smooth CoM velocity, angular velocity, and GRFs
• **Swing foot cost** to keep the replanned footstep close to the nominal target
• **Lateral clearance penalty** to avoid leg cross-over and self-collisions

Roll and pitch are strongly penalized to prevent falling, while yaw is tracked more softly to allow natural torso rotation during locomotion and disturbance recovery.

---

## 🧱 **Physical and Contact Constraints**

The optimization problem enforces physical feasibility through several constraints:

• SRBD dynamic consistency
• Friction cone constraints
• Unilateral contact constraints
• GRF bounds
• Angular velocity limits
• Step length limits
• Lateral clearance constraints

The friction cone approximation is expressed as:

```math
|f_x| \leq \mu f_z, \qquad |f_y| \leq \mu f_z
```

while unilateral contact is enforced by:

```math
f_z \geq 0
```

This guarantees that the ground can push the robot but cannot pull it.

---

## 🦿 **Whole-Body Control Layer**

The **Whole-Body Controller (WBC)** acts as a bridge between the reduced-order SRBD-MPC model and the full humanoid robot.

It solves an **inverse dynamics Quadratic Program (QP)** to compute:

• joint accelerations
• contact forces
• motor torques

The full-body dynamics are imposed as:

```math
M(q)\ddot{q}+h(q,\dot{q})=S^T\tau+J_c^Tf_c
```

where:

• `M(q)` is the full-body mass matrix
• `h(q,q_dot)` contains gravity, Coriolis, and centrifugal terms
• `tau` are the actuated joint torques
• `f_c` are the contact forces
• `S` is the selection matrix for actuated joints
• `J_c` is the contact Jacobian

The WBC tracks CoM, torso orientation, swing foot motion, and MPC contact force references while respecting full-body physical constraints.

---

## 🛡️ **Safety and Robustness Features**

Several additional mechanisms are included to improve robustness:

### **Knee Anti-Hyperextension Guard**

Prevents the knee from reaching dangerous hyperextended configurations by imposing acceleration bounds inside the WBC-QP.

### **Lateral Clearance Penalty**

Avoids leg cross-over and self-collisions during aggressive recovery steps.

### **Dynamic Cost Weight Scheduling**

Temporarily relaxes attitude tracking during severe disturbances, allowing the torso to absorb part of the impact before returning to the nominal posture.

### **Arm Swinging Heuristic**

Commands shoulder pitch motion opposite to leg displacement:

```math
\theta_{arm}=-k_{swing}\Delta p_{leg}
```

This reduces parasitic yaw rotations and improves natural walking behavior without adding a full angular momentum optimization inside the MPC.

### **Safe Fallback Strategy**

If the nonlinear solver fails to converge, the controller distributes the robot weight across active contact points to prevent immediate collapse.

---

## 🔬 **Simulation Scenarios**

The framework is validated in simulation on the **HRP-4 humanoid robot** using **DARTpy**.

The main tests include:

### 🟢 **Nominal Walking**

The robot follows the planned gait without external disturbances, validating the basic SRBD-MPC and WBC pipeline.

### 🟡 **Lateral Push in Double Support**

A lateral force is applied while both feet are in contact with the ground. The robot uses the larger support polygon and GRF redistribution to recover stability.

### 🟠 **Lateral Push in Single Support**

The same disturbance is applied during a more critical phase, when only one foot is supporting the robot. The MPC adapts the next footstep to recover balance.

### 🔴 **Limit-Breaking Tests**

Different controller configurations are tested under increasing disturbance intensities to evaluate the robustness limits of the approach.

---

## ⚙️ Technologies

| Category             | Tools / Topics                                                  |
| -------------------- | --------------------------------------------------------------- |
| Programming          | Python                                                          |
| Optimization         | CasADi, IPOPT                                                   |
| Simulation           | DARTpy                                                          |
| Scientific Computing | NumPy, Matplotlib                                               |
| Robotics             | Humanoid Robotics, Model Predictive Control, Whole-Body Control |
| Methods              | Nonlinear Optimization, Inverse Dynamics, Push Recovery         |
---

## 🎯 **Main Contributions**

This project demonstrates that combining **SRBD-MPC**, **multi-contact GRF optimization**, **reactive footstep replanning**, and **whole-body inverse dynamics control** enables a humanoid robot to walk robustly and recover from lateral disturbances without relying solely on simplified LIP-based gait generation.

The framework highlights the importance of rotational dynamics, angular momentum regulation, contact force distribution, and adaptive foothold selection for robust humanoid locomotion.
