import dartpy as dart
import os

world = dart.simulation.World()
urdfParser = dart.utils.DartLoader()
current_dir = os.path.dirname(os.path.abspath('simulation.py'))
hrp4   = urdfParser.parseSkeleton(os.path.join(current_dir, "urdf", "hrp4.urdf"))
print("DoF names:")
for i in range(hrp4.getNumDofs()):
    print(f"{i}: {hrp4.getDof(i).getName()}")
