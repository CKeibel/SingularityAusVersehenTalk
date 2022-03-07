import gym
from time import sleep
from wrappers import make_env
import tensorflow as tf
import numpy as np

AGENT_LIST = ["suicide_pacman", "good_pacman"]
GAME_LIST = ["MsPacman-v0", "Pong-v0"]
EPISODES = 2


def load_agent(name: str):
    PATH = f"model/{name}"
    agent = tf.keras.models.load_model(PATH)
    return agent


if __name__ == "__main__":
    agent = load_agent(AGENT_LIST[1])
    env = make_env(GAME_LIST[0])

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(agent.predict(state))
            try:
                state, reward, done, info = env.step(action)
            except:
                state, reward, done, info = env.step(env.action_space.sample())
            sleep(0.1)
            env.render()
