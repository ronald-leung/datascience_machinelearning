# random agent
def random_agent_opposite (observation, configuration):
    import numpy as np
    shoot = np.random.randint(0,3)
    #Reverse the shoot.
    if shoot == 0:
        return 1
    elif shoot == 1:
        return 2
    else:
        return 0
