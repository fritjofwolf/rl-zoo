import tensorflow as tf
import gym
import numpy as np
from src.data_collection import collect_data
import numpy as np

class VPG():

    def __init__(self, env):
        self._env = env
        if type(env.action_space) == 'gym.spaces.box.Box':
            self._build_computational_graph_continuous_actions()
        else:
            self._build_computational_graph_categoriacal_actions()   


    def train(self, policy_learning_rate=0.0003, value_function_learning_rate = 0.001, value_function_updates = 10,\
              n_epochs = 10):
        obs_ph, act_ph, weights_ph, actions, state_values, policy_loss, state_value_loss = self._graph

        optimizer_policy = tf.train.AdamOptimizer(policy_learning_rate)
        train_policy = optimizer_policy.minimize(policy_loss)
        optimizer_state_value = tf.train.AdamOptimizer(value_function_learning_rate)
        train_state_value = optimizer_state_value.minimize(state_value_loss)

        sess = tf.Session()
        episode_returns = []

        sess.run(tf.global_variables_initializer())
        for i in range(n_epochs):
            tmp1, tmp2, tmp3, batch_rets, batch_len = collect_data(self._env, sess, self._graph, 4000 , render = False)
            episode_returns.extend(batch_rets)
            print(i, np.mean(batch_rets), np.min(batch_rets), np.max(batch_rets))
            sess.run([train_policy],feed_dict={
                                            obs_ph: np.array(tmp1),
                                            act_ph: np.array(tmp2),
                                            weights_ph: np.array(tmp3)
                                        })
            for _ in range(10):
                sess.run([train_state_value],feed_dict={
                                        obs_ph: np.array(tmp1),
                                        act_ph: np.array(tmp2),
                                        weights_ph: np.array(tmp3)
                                    })
            # v = sess.run([state_values], feed_dict={
            #                                 obs_ph: np.array(tmp1),
            #                                 act_ph: np.array(tmp2),
            #                                 weights_ph: np.array(tmp3)
            #                             })
            # print("Value function mean and std:", np.mean(v), np.std(v))
            #print('Optimized')
        return episode_returns


    def _build_computational_graph_categorical_actions(self):
        env = self._env
        obs_dim = env.observation_space.shape[0]
        n_acts = env.action_space.n

        # placeholder
        obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
        act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
        weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)

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

        # make action selection op (outputs int actions, sampled from policy)
        actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)

        # make loss function whose gradient, for the right data, is policy gradient
        action_masks = tf.one_hot(act_ph, n_acts)
        log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
        policy_loss = -tf.reduce_mean((weights_ph - state_values) * log_probs)

        state_value_loss = tf.reduce_mean((weights_ph - state_values)**2)
        graph = [obs_ph, act_ph, weights_ph, actions, state_values, policy_loss, state_value_loss]
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
        # compute actions
        actions = tf.random.normal((1,1), mean=means, stddev=std)

        # make loss function whose gradient, for the right data, is policy gradient
        first_summand = tf.reduce_sum(((act_ph - means) / std)**2 + 2*log_std)
        log_probs = -0.5*(first_summand + n_acts * tf.math.log(2*np.pi))
        policy_loss = -tf.reduce_mean((weights_ph - state_values) * log_probs)

        state_value_loss = tf.reduce_mean((weights_ph - state_values)**2)
        graph = [obs_ph, act_ph, weights_ph, actions, state_values, policy_loss, state_value_loss]
        self._graph = graph