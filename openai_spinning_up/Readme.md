# Collection of some Deep Reinforcement Learning Algorithms

## Summary

The set of algorithms corresponds to the choice suggested by OpenAI Spinning Up course (https://spinningup.openai.com/en/latest/). Namely the following algorithms were implemented:

1. Vanilla Policy Gradient (aka REINFORCE)
2. Deep-Q-Network
3. Synchronous Actor-Critic Methods
4. Proximal Policy Optimization
5. Deep Deterministic Policy Gradient

## Results

Following, there are some results from the implementation of the Proximal Policy Optimization algorithm on two well-known test enviroments for reinforcement learning from the OpenAI Gym Enviroment.

### LunarLander-v2

LunarLander-v2 (https://gym.openai.com/envs/LunarLander-v2/) is a simple enviroment with categorical actions. The goal is to land a lander onto the landing pad without crashing it by using three different fire engines while using as less fuel as possible. With 6 state variables, 4 different actions and a time horizon of 1000 steps, it is much harder to master than CartPole, but still easy enough to do several iterations in a reasonable amount of time.

![Alt text](openai_spinning_up/media/lunar_lander_2000.jpg?raw=true "Average Performance for LunarLander-v2")

### Half-Cheetah-v0