import tensorflow as tf
import gym
import numpy as np
import logging

from ..data_collection.a2c_data_collection import A2CDataCollector

class PPO():

    def __init__(self, env_name, gamma = 0.9, learning_rate = 0.0003):
        self._env = gym.make(env_name)
        self._learning_rate = learning_rate
        self._env_name = env_name
        self._gamma = gamma
        self._obs_dim = self._env.observation_space.shape[0]
        if type(env.action_space) == gym.spaces.box.Box:
            self._build_computational_graph_continuous_actions()
            self._n_acts = self._env.action_space.shape[0]
        else:
            self._build_computational_graph_categorical_actions()
            self._n_acts = self._env.action_space.n


    def train(self, n_epochs):
        [obs_ph, act_ph, new_obs_ph, rew_ph, terminal_ph, policy_network, old_policy_network, actions] = self._graph
        data_collector = DataCollector(sess, self._env_name, actions, obs_ph, 20, 50)
        for i in range(n_epochs):
            self._update_old_network()
            obs, acts, new_obs, rews, terminal = data_collector.collect_data()
            data_collector.print_return_statistics()
            for j in range(K):
                print(sess.run([max_r,max_advantages], feed_dict ={
                        obs_ph: np.array(obs).reshape(-1, obs_dim),
                        act_ph: np.array(acts),
                        new_obs_ph: np.array(new_obs).reshape(-1, obs_dim),
                        rew_ph: np.array(rews).reshape(-1, 1),
                        terminal_ph: np.array(terminal).reshape(-1, 1)
                }))
                sess.run([train_policy],feed_dict={
                                            obs_ph: np.array(obs).reshape(-1, obs_dim),
                                            act_ph: np.array(acts),
                                            new_obs_ph: np.array(new_obs).reshape(-1, obs_dim),
                                            rew_ph: np.array(rews).reshape(-1, 1),
                                            terminal_ph: np.array(terminal).reshape(-1, 1)
                                        })
            for j in range(30):
                sess.run([train_state_value],feed_dict={
                                            obs_ph: np.array(obs).reshape(-1, obs_dim),
                                            act_ph: np.array(acts),
                                            new_obs_ph: np.array(new_obs).reshape(-1, obs_dim),
                                            rew_ph: np.array(rews).reshape(-1, 1),
                                            terminal_ph: np.array(terminal).reshape(-1, 1)
                                        })
            print('State value loss is:')
            print(sess.run(state_value_loss, feed_dict ={
                        obs_ph: np.array(obs).reshape(-1, obs_dim),
                        act_ph: np.array(acts),
                        new_obs_ph: np.array(new_obs).reshape(-1, obs_dim),
                        rew_ph: np.array(rews).reshape(-1, 1),
                        terminal_ph: np.array(terminal).reshape(-1, 1)
                }))
            print()

    def _update_old_network(self):
        policy_network = self._graph[5]
        old_policy_network = self._graph[6]
        self._sess.run([v_t.assign(v) for v_t, v in zip(old_policy_network.trainable_weights, policy_network.trainable_weights)])

    def _build_computational_graph_categorical_actions(self):
        # define placeholder
        obs_ph = tf.placeholder(shape=(None, self._obs_dim), dtype=tf.float32)
        act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
        new_obs_ph = tf.placeholder(shape=(None, self._obs_dim), dtype=tf.float32)
        rew_ph = tf.placeholder(shape=(None,1), dtype=tf.float32)
        terminal_ph = tf.placeholder(shape=(None,1), dtype=tf.float32)

        # build networks
        policy_network = self._build_network('tanh', self._n_acts)
        old_policy_network = self._build_network('tanh', self._n_acts)
        state_value_network = self._build_network('relu', 1)
        
        state_value = state_value_network(obs_ph)
        new_state_value = state_value_network(new_obs_ph)
        td_target = rew_ph + gamma * new_state_value * (1-terminal_ph)

        # make loss function whose gradient, for the right data, is policy gradient
        obs_logits = policy_network(obs_ph)
        obs_logits_old_network = policy_network(obs_ph)
        actions = tf.squeeze(tf.multinomial(logits=obs_logits,num_samples=1), axis=1)
        action_masks = tf.one_hot(act_ph, self._n_acts)
        selected_action_probs = tf.reduce_sum(action_masks * tf.nn.softmax(obs_logits), axis=1)
        selected_action_probs_old_network = tf.reduce_sum(action_masks * tf.nn.softmax(obs_logits_old_network), axis=1)

        r = selected_action_probs / tf.stop_gradient(selected_action_probs_old_network)
        advantages = tf.squeeze(td_target - state_value, axis=1)
        factor = 1 + 0.2 * tf.math.sign(advantages)
        x = tf.math.minimum(advantages*r, advantages*factor)
        policy_loss = -tf.reduce_mean(x)

        state_value_loss = tf.losses.mean_squared_error(y, state_value)

        policy_optimizer = tf.train.AdamOptimizer(self._learning_rate)
        state_value_optimizer = tf.train.AdamOptimizer(self._learning_rate)
        train_policy = policy_optimizer.minimize(policy_loss)
        train_state_value = state_value_optimizer.minimize(state_value_loss)
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        self._graph = [obs_ph, act_ph, new_obs_ph, rew_ph, terminal_ph, \
                        policy_network, old_policy_network, actions]


    def _build_network(self, activation = 'relu', n_output_units = 1):
        mlp = tf.keras.models.Sequential()
        mlp.add(tf.keras.layers.Dense(50, activation=activation))
        mlp.add(tf.keras.layers.Dense(50, activation=activation))
        mlp.add(tf.keras.layers.Dense(n_output_units))
        return mlp