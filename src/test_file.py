from dqn import DQN
import numpy as np
import tkinter
import matplotlib.pyplot as plt

dqn_model = DQN('CartPole-v0')
results = dqn_model.train(1000)
print(results)
plt.plot([x[0] for x in results], [x[1] for x in results])