import matplotlib.pyplot as plt
import numpy as np

def visualize_episode_return(episode_returns, colors, smoothing_window = 100, title=''):
    n_runs = len(episode_returns)
    episode_mean_returns = [[np.mean(episode_returns[j][i-smoothing_window:i]) for i in range(smoothing_window, len(episode_returns[j]))] \
                                for j in range(n_runs)]
    # Episode return
    fig = plt.figure(figsize=(12,12))
    for i in range(n_runs):
        plt.plot(range(smoothing_window,len(episode_returns[i])), episode_mean_returns[i], color=colors[i])
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Episode Return (smoothed over time window of ' + str(smoothing_window) + ' steps)')
    plt.title(title)


def visualize_epoch_state_value_loss(epoch_state_value_loss, colors, smoothing_window = 20, title=''):
    n_runs = len(epoch_state_value_loss)
    mean_epoch_state_value_loss = [[np.mean(epoch_state_value_loss[j][i-smoothing_window:i]) 
                                for i in range(smoothing_window, len(epoch_state_value_loss[j]))] for j in range(n_runs)]
    # Epoch loss
    fig2 = plt.figure(figsize=(12,12))
    for i in range(n_runs):
        plt.plot(range(smoothing_window,len(epoch_state_value_loss[i])), mean_epoch_state_value_loss[i], color=colors[i])
    plt.xlabel('Epochs')
    plt.ylabel('Epoch State Value Loss (smoothed over time window of ' + str(smoothing_window) + ' steps)')
    plt.title(title)


def visualize_epoch_entropy(epoch_entropy, colors, smoothing_window = 10, title=''):
    n_runs = len(epoch_entropy)
    mean_epoch_entropy = [[np.mean(epoch_entropy[j][i-smoothing_window:i]) 
                                for i in range(smoothing_window, len(epoch_entropy[j]))] for j in range(n_runs)]
    # Epoch loss
    fig3 = plt.figure(figsize=(12,12))
    for i in range(n_runs):
        plt.plot(range(smoothing_window,len(epoch_entropy[i])), mean_epoch_entropy[i], color=colors[i])
    plt.xlabel('Epochs')
    plt.ylabel('Epoch Entropy (smoothed over time window of ' + str(smoothing_window) + ' steps)')
    plt.title(title)
