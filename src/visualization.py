import matplotlib.pyplot as plt
%matplotlib inline

def visualize_episode_return(episode_returns, smoothing_window = 100, title=''):
    n_runs = len(episode_returns)
    episode_mean_returns = [[np.mean(episode_returns[j][i-smoothing_window:i]) for i in range(smoothing_window, len(episode_returns[j]))] \
                                for j in range(n_runs)]
    fig = plt.figure(figsize=(12,12))
    for i in range(n_runs):
        plt.plot(range(smoothing_window,len(episode_mean_returns[i])+smoothing_window), episode_mean_returns[i], color=np.random.rand(3))
    plt.xlabel('Episodes')
    plt.ylabel('Mean Episode Return (smoothed over time window of' + smoothing_window + 'steps)')
    plt.title(title)