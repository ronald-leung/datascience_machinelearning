# random agent
def circle_agent (observation, configuration):    
    pi = "141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128481117450284102701938521105559644622948954930381964428810975665933446128475648233"
    
    if observation.step < 0:
        index = 0
    elif observation.step >= len(pi):
        index = observation.step % len(pi)
    else:
        index = observation.step
    shoot = int(pi[index]) % 3

    return shoot