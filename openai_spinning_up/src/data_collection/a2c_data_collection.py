import gym
import numpy as np

class A2CDataCollector():
    
    def __init__(self, sess, env_name, actions, obs_ph, n_actors, n_samples):
        self._sess = sess
        self._envs = [gym.make(env_name) for _ in range(n_actors)]
        self._states = [env.reset() for env in self._envs]
        self._actor_rews = [[] for _ in range(n_actors)]
        self._n_samples = n_samples
        self._actions = actions
        self._obs_ph = obs_ph
        self._returns = []
        self._lens = []
        
    def collect_data(self):
        batch_obs = []
        batch_acts = []
        batch_new_obs = []
        batch_rews = []
        batch_terminal = []
        for i in range(len(self._envs)):
            tmp_data = self._collect_data_single_actor(i)
            batch_obs.extend(tmp_data[0])
            batch_acts.extend(tmp_data[1])            
            batch_new_obs.extend(tmp_data[2])            
            batch_rews.extend(tmp_data[3])
            batch_terminal.extend(tmp_data[4])
        return batch_obs, batch_acts, batch_new_obs, batch_rews, batch_terminal  
    
    def print_return_statistics(self):
        print('Statistics of the last 100 episodes:')
        ret_mean = np.mean(self._returns[-100:])
        ret_std = np.std(self._returns[-100:])
        ret_min = np.min(self._returns[-100:])
        ret_max = np.max(self._returns[-100:])
        print(ret_mean, ret_std, ret_min, ret_max)
    
    def get_episode_statistics(self):
        return self._returns

    def _collect_data_single_actor(self, env_id):
        batch_obs = []
        batch_acts = []
        batch_new_obs = []
        batch_rews = []
        batch_terminal = []
        
        env = self._envs[env_id]
        obs = self._states[env_id]
        done = False
        
        for _ in range(self._n_samples):
            batch_obs.append(obs.copy())
            act = self._sess.run(self._actions, {self._obs_ph: obs.reshape(1,-1)})[0]
            obs, rew, done, info = env.step(act)
            batch_new_obs.append(obs.copy())
            batch_terminal.append(float(done))
            batch_acts.append(act)
            batch_rews.append(rew)
            self._actor_rews[env_id].append(rew)
            if done:
                ep_len = len(self._actor_rews[env_id])
                ep_ret = sum(self._actor_rews[env_id])
                self._returns.append(ep_ret)
                self._lens.append(ep_len)
                self._actor_rews[env_id] = []
                
#                 if ep_len == 200:
#                     batch_terminal[-1] = 0.0
                obs, done= env.reset(), False
        self._states[env_id] = obs
        return batch_obs, batch_acts, batch_new_obs, batch_rews, batch_terminal