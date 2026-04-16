import dartpy as dart
import os, sys, copy
sys.path.append(os.path.join(os.getcwd(), 'ismpc'))
from simulation import Hrp4Controller

world = dart.simulation.World()
urdfParser = dart.utils.DartLoader()
current_dir = os.path.join(os.getcwd(), "ismpc")
hrp4   = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "hrp4.urdf"))
ground = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "ground.urdf"))
world.addSkeleton(hrp4)
world.addSkeleton(ground)
world.setGravity([0, 0, -9.81])
world.setTimeStep(0.01)

import numpy as np
default_inertia = dart.dynamics.Inertia(1e-8, np.zeros(3), 1e-10 * np.identity(3))
for body in hrp4.getBodyNodes():
    if body.getMass() == 0.0:
        body.setMass(1e-8)
        body.setInertia(default_inertia)

node = Hrp4Controller(world, hrp4)
node.save_plots_as_images = False
node.show_plots_interactive = False

try:
    for i in range(2000): # 20 seconds
        node.customPreStep()
        world.step()
        if node.shutdown_triggered:
            break
except SystemExit as e:
    pass
print("Done headless.")
