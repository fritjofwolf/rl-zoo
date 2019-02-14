import tensorflow as tf
import gym
import numpy as np
from src.data_collection import collect_data
import numpy as np
import logging

class DQN():

    def __init__(self, env, network_type = 'mlp'):
        self._env = env
        self._build_computational_graph_categorical_actions(network_type)


    def train(self, n_episodes):
        row_pointer = 0
        learnable = False
        for i in range(n_episodes):
            if i % 100 == 0:
                sess.run([v_t.assign(v) for v_t, v in zip(mlp_target.trainable_weights, mlp_action_val.trainable_weights)])
            done = False
            state = env.reset()
            cnt = 0
            while not done:
                action = select_action(sess, state, 0.1)
                new_state, reward, done, _ = env.step(action)
                add_sample(experience_replay_buffer, state, action, reward, new_state, done, row_pointer)
                row_pointer += 1
                if row_pointer == exp_replay_size:
                    learnable = True
                row_pointer %= exp_replay_size
                state = new_state
                if learnable:
                    update_q_function()
                cnt += 1

            if i % 100 == 0:
                print('Evaluation after',i,'steps:',evaluate(sess, env, 10))


    def preprocess_atari_state(self, state):
        pass


    def _build_computational_graph_categorical_actions(self, network_type):
        env = self._env
        obs_dim = env.observation_space.shape[0]
        n_acts = env.action_space.n
        gamma = 0.9

        obs_ph = tf.placeholder(shape=(None,obs_dim), dtype=tf.float32)
        act_ph = tf.placeholder(shape=(None), dtype=tf.int32)
        rew_ph = tf.placeholder(shape=(None), dtype=tf.float32)
        new_obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
        terminal_ph = tf.placeholder(shape=(None), dtype=tf.float32)

        # make core of state-action-value function network
        if network_type == 'mlp':
            q_value_network = self._build_mlp()
            target_network = self._build_mlp()
        else:
            q_value_network = self._build_cnn()
            target_network = self._build_cnn()

        # define state action values
        old_state_action_values = q_value_network(obs_ph)
        new_state_action_values = target_network(new_obs_ph)

        # select action
        greedy_action = tf.math.argmax(old_state_action_values, axis=1)

        # define loss function
        y = rew_ph + gamma * tf.reduce_max(new_state_action_values, axis=1)*(1-terminal_ph)
        action_masks = tf.one_hot(act_ph, n_acts)
        old_selected_action_values = tf.reduce_sum(action_masks * old_state_action_values, axis=1)
        action_value_loss = tf.losses.mean_squared_error(tf.stop_gradient(y), old_selected_action_values)

        # define optimizer
        optimizer_action_value = tf.train.AdamOptimizer(0.001)
        train_action_value = optimizer_action_value.minimize(action_value_loss)

        self._graph = [obs_ph, act_ph, rew_ph, new_obs_ph, terminal_ph, greedy_action,\
                 state_values, entropy, train_action_value, action_value_loss]


    def _build_mlp(self):
        q_value_network = tf.keras.models.Sequential()
        q_value_network.add(tf.keras.layers.Dense(50, activation='relu'))
        q_value_network.add(tf.keras.layers.Dense(50, activation='relu'))
        q_value_network.add(tf.keras.layers.Dense(n_acts, activation=None))
        return q_value_network

    def _build_cnn(self):
        q_value_network = tf.keras.models.Sequential()
        # todo add cnn

        return q_value_network