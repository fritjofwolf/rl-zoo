import time

def collect_data(env, sess, computational_graph, batch_size, gamma = 0.99, render=False):
    obs_ph, act_ph, weights_ph, actions, state_values, loss, state_value_loss = computational_graph
    # make some empty lists for logging.
    batch_obs = []          # for observations
    batch_acts = []         # for actions
    batch_weights = []      # for R(tau) weighting in policy gradient
    batch_rets = []         # for measuring episode returns
    batch_lens = []         # for measuring episode lengths

    # reset episode-specific variables
    obs = env.reset()       # first obs comes from starting distribution
    done = False            # signal from environment that episode is over
    ep_rews = []            # list for rewards accrued throughout ep
    # collect experience by acting in the environment with current policy
    while True:
        if render:
            env.render()
            time.sleep(0.01)
        
        batch_obs.append(obs.copy())

        # act in the environment
        act = sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]
        obs, rew, done, _ = env.step(act)
        
        # save action, reward
        batch_acts.append(act)
        ep_rews.append(rew)

        if done:
            if render:
                env.close()
                render = False
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            
            if ep_len == env._max_episode_steps:
                bootstrap_value = sess.run(state_values, {obs_ph:obs.reshape(1,-1)})[0][0]
            else:
                bootstrap_value = 0
            batch_weights += compute_rewards_to_go(ep_rews, gamma, bootstrap_value)
            
            # reset episode-specific variables
            obs, done, ep_rews = env.reset(), False, []

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break
    return batch_obs, batch_acts, batch_weights, batch_rets, batch_lens

def compute_rewards_to_go(rewards, gamma, bootstrap_value):
    rewards_to_go = [rewards[-1] + gamma*bootstrap_value]
    for rew in rewards[:-1][::-1]:
        tmp = rewards_to_go[-1]
        rewards_to_go.append(rew + gamma * tmp)
    return rewards_to_go[::-1]