# I assume you will try to win, from last match. 
def assume_you_try_to_beat_my_last_agent (observation, configuration):
    if observation.step > 0:
        if observation.lastOpponentAction == 0:            
            return 2
        elif observation.lastOpponentAction == 1:
            return 0
        else:
            return 1
    else:
        return 0