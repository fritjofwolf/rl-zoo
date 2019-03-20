import tensorflow as tf
import gym
import numpy as np
import logging
import tqdm

from ..data_collection.a2c_data_collection import A2CDataCollector

class A2C():

    def __init__(self, env, env_name, gamma = 0.9):
        self._env = env
        self._env_name = env_name
        self._gamma = gamma
        self._obs_dim = self._env.observation_space.shape[0]
        if type(env.action_space) == gym.spaces.box.Box:
            self._n_acts = self._env.action_space.shape[0]
            self._build_computational_graph_continuous_actions()
        else:
            self._n_acts = self._env.action_space.n
            self._build_computational_graph_categorical_actions()   


    def train(self, policy_learning_rate=0.0003, value_function_learning_rate = 0.001, n_value_function_updates = 10,\
              n_epochs = 10):
        obs_ph, act_ph, new_obs_ph, rew_ph, terminal_ph, actions, state_values, entropy, policy_loss, state_value_loss, x = self._graph

        optimizer_policy = tf.train.AdamOptimizer(policy_learning_rate)
        train_policy = optimizer_policy.minimize(policy_loss)
        optimizer_state_value = tf.train.AdamOptimizer(value_function_learning_rate)
        train_state_value = optimizer_state_value.minimize(state_value_loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        data_collector = A2CDataCollector(sess, self._env_name, actions, obs_ph, 20, 50)
        
        for i in tqdm.tqdm_notebook(range(n_epochs)):
            obs, acts, new_obs, rews, terminal = data_collector.collect_data()
            # print(np.array(sess.run(x, feed_dict = {
            #                                 obs_ph: np.array(obs).reshape(-1, self._obs_dim),
            #                                 act_ph: np.array(acts),
            #                                 new_obs_ph: np.array(new_obs).reshape(-1, self._obs_dim),
            #                                 rew_ph: np.array(rews).reshape(-1, 1),
            #                                 terminal_ph: np.array(terminal).reshape(-1, 1)
            # })).shape)
            sess.run([train_policy],feed_dict={
                                            obs_ph: np.array(obs).reshape(-1, self._obs_dim),
                                            act_ph: np.array(acts),
                                            new_obs_ph: np.array(new_obs).reshape(-1, self._obs_dim),
                                            rew_ph: np.array(rews).reshape(-1, 1),
                                            terminal_ph: np.array(terminal).reshape(-1, 1)
                                        })
            for _ in range(n_value_function_updates):
                sess.run([train_state_value],feed_dict={
                                        obs_ph: np.array(obs).reshape(-1, self._obs_dim),
                                        act_ph: np.array(acts),
                                        new_obs_ph: np.array(new_obs).reshape(-1, self._obs_dim),
                                        rew_ph: np.array(rews).reshape(-1, 1),
                                        terminal_ph: np.array(terminal).reshape(-1, 1)
                                    })
        return data_collector.get_episode_statistics()


    def _build_computational_graph_categorical_actions(self):
        env = self._env
        obs_dim = env.observation_space.shape[0]
        n_acts = env.action_space.n

        # placeholder
        obs_ph = tf.placeholder(shape=(None, self._obs_dim), dtype=tf.float32)
        act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
        new_obs_ph = tf.placeholder(shape=(None, self._obs_dim), dtype=tf.float32)
        rew_ph = tf.placeholder(shape=(None,1), dtype=tf.float32)
        terminal_ph = tf.placeholder(shape=(None,1), dtype=tf.float32)

        # make core of policy network
        mlp = tf.keras.models.Sequential()
        mlp.add(tf.keras.layers.Dense(30, activation='tanh'))
        mlp.add(tf.keras.layers.Dense(30, activation='tanh'))
        mlp.add(tf.keras.layers.Dense(n_acts))
        logits = mlp(obs_ph)

        # value function network
        state_value_mlp = tf.keras.models.Sequential()
        state_value_mlp.add(tf.keras.layers.Dense(50, activation='relu'))
        state_value_mlp.add(tf.keras.layers.Dense(50, activation='relu'))
        state_value_mlp.add(tf.keras.layers.Dense(1))
        state_values = state_value_mlp(obs_ph)
        new_state_values = state_value_mlp(new_obs_ph)
        td_target = rew_ph + self._gamma * new_state_values * (1-terminal_ph)

        # make action selection op (outputs int actions, sampled from policy)
        actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)
        action_probs = tf.nn.softmax(logits)
        entropy = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs), axis=1))

        # make loss function whose gradient, for the right data, is policy gradient
        action_masks = tf.one_hot(act_ph, n_acts)
        log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
        x = tf.squeeze(td_target - state_values, axis=1) * log_probs
        policy_loss = -tf.reduce_mean(x)

        y = (td_target - state_values)**2
        state_value_loss = tf.reduce_mean(y)

        graph = [obs_ph, act_ph, new_obs_ph, rew_ph, terminal_ph, actions, state_values, entropy, policy_loss, state_value_loss, x]
        self._graph = graph



    def _build_computational_graph_continuous_actions(self):
        env = self._env
        obs_dim = env.observation_space.shape[0]
        n_acts = env.action_space.shape[0]

        # placeholder
        obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
        act_ph = tf.placeholder(shape=(None,n_acts), dtype=tf.float32)
        weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)

        # network for gaussian means
        mlp = tf.keras.models.Sequential()
        mlp.add(tf.keras.layers.Dense(50, activation='tanh'))
        mlp.add(tf.keras.layers.Dense(50, activation='tanh'))
        mlp.add(tf.keras.layers.Dense(n_acts))
        means = mlp(obs_ph)

        # value function network
        state_value_mlp = tf.keras.models.Sequential()
        state_value_mlp.add(tf.keras.layers.Dense(50, activation='relu'))
        state_value_mlp.add(tf.keras.layers.Dense(50, activation='relu'))
        state_value_mlp.add(tf.keras.layers.Dense(1))
        state_values = state_value_mlp(obs_ph)

        # variances
        log_std = tf.Variable(-0.5)
        std = tf.math.exp(log_std)
        entropy = 0.5 * tf.math.log(2*np.pi*np.e) + 0.5*log_std
        # compute actions
        actions = tf.random.normal((1,1), mean=means, stddev=std)


        # make policy loss function whose gradient, for the right data, is policy gradient
        first_summand = tf.reduce_sum(((act_ph - means) / std)**2 + 2*log_std)
        log_probs = -0.5*(first_summand + n_acts * tf.math.log(2*np.pi))
        policy_loss = -tf.reduce_mean((weights_ph - state_values) * log_probs)

        state_value_loss = tf.reduce_mean((weights_ph - state_values)**2)

        graph = [obs_ph, act_ph, weights_ph, actions, state_values, entropy, policy_loss, state_value_loss]
        self._graph = graph