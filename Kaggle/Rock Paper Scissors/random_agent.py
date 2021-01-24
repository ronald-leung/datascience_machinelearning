# random agent
def random_agent (observation, configuration):
    import numpy as np
#     if observation.step > 0:
#         print("Step: ", observation.step, " Observation last Opponent Action: ", observation.lastOpponentAction)    
    shoot = np.random.randint(0,3)
#     print("My shot : ", shoot)
#     print("\n")
    return shoot