from Auxiliaries import evaluate_policy
from Policies import ProBSP
from NewEnvironment import Inventory, GeometricOrderPipeline

pipeline = GeometricOrderPipeline(100, 0.2)
env = Inventory(50,pipeline, 10, 5, 1,emergency_cost=10)
probsp = ProBSP(env, 20, 15, 10)

evaluate_policy(env, probsp, replication=12, processors=4)