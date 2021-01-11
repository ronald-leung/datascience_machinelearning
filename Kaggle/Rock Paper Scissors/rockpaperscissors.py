class Agent():
    def my_random_agent(observation, configuration):
        import numpy as np
        shoot = np.random.randint(1,4)
        return shoot

agent = Agent()


def call_agent(observation, configuration):
    return agent(observation, configuration)

