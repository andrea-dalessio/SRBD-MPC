import sys, os
sys.path.append(os.path.join(os.getcwd(), 'ismpc'))
from simulation import Hrp4Controller
from footstep_planner import FootstepPlanner
params = {'g': 9.8, 'foot_size': 0.1, 'first_swing': 'rfoot', 'ss_duration': 70, 'ds_duration': 30}
params['initial_stationary_steps'] = 2
params['first_step_ds_multiplier'] = 2
params['stationary_step_ds_duration'] = 100
initial = {'lfoot': {'pos':[0,0,0,0,0.1,0]}, 'rfoot': {'pos':[0,0,0,0,-0.1,0]}}
planner = FootstepPlanner([(0.1, 0, 0.2)]*5, initial['lfoot']['pos'], initial['rfoot']['pos'], params)

for t in range(0, 300, 10):
    idx = planner.get_step_index_at_time(t)
    phase = planner.get_phase_at_time(t)
    print(f"t={t}: phase={phase}, idx={idx}, foot={planner.plan[idx]['foot_id'] if idx is not None else None}")
