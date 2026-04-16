import dartpy as dart
import os, sys
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

node = Hrp4Controller(world, hrp4)
node.save_plots_as_images = False
node.show_plots_interactive = False
os.environ['DISTURBANCE_START_S'] = '4.80'
os.environ['DISTURBANCE_END_S'] = '4.95'
for i in range(700): # 7 seconds
    node.customPreStep()
    world.step()
    if node.shutdown_triggered:
        print(f"Crased at tick {i}")
        break
