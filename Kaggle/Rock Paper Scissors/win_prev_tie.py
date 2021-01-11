# Whatever you do now, I'll try to shoot the winning option the next
def win_prev_tie_agent (observation, configuration):
    if observation.step > 0:
        if observation.lastOpponentAction == 0:
            return 1
        elif observation.lastOpponentAction == 1:
            return 2
        else:
            return 0
    else:
        return 0