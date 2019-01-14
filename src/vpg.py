import tensorflow as tf
import gym
import numpy as np

class VPG():

    def __init__(self, env):
        self._env = env

    def train(self, policy_learning_rate, value_function_learning_rate, value_function_updates, n_epochs = 100, n_runs = 10):
        # todo graph laden
        optimizer = tf.train.AdamOptimizer(0.003)
        # optimizer_action_value = tf.train.AdamOptimizer(0.001)
        optimizer_state_value = tf.train.AdamOptimizer(0.001)
        train = optimizer.minimize(loss)
        # train_action_value = optimizer_action_value.minimize(loss_action_value)
        train_state_value = optimizer_state_value.minimize(loss_state_value)

        sess = tf.Session()
        n_runs = 3
        n_epochs = 300
        episode_returns = [[] for x in range(n_runs)]

        for j in range(n_runs):
            sess.run(tf.global_variables_initializer())
            for i in range(n_epochs):
                tmp1, tmp2, tmp3, batch_rets, batch_len = collect_data(sess, 4000 ,debug=False)
                episode_returns[j].extend(batch_rets)
                print(i, np.mean(batch_rets), np.min(batch_rets), np.max(batch_rets))
                sess.run([train],feed_dict={
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
                v = sess.run([state_value], feed_dict={
                                                obs_ph: np.array(tmp1),
                                                act_ph: np.array(tmp2),
                                                weights_ph: np.array(tmp3)
                                            })
                print("Value function mean and std:", np.mean(v), np.std(v))
                #print('Optimized')
            print('Evaluation')
            tmp1, tmp2, tmp3, batch_rets, batch_len = collect_data(sess, 5, debug=True)
            print(np.mean(batch_len), np.min(batch_len), np.max(batch_len))
            print()
        return episode_returns


    def build_computational_graph(self):
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
        loss = -tf.reduce_mean((weights_ph - state_values) * log_probs)

        state_value_loss = tf.reduce_mean((weights_ph - state_values)**2)
        graph = {obs_ph, act_ph, weights_ph, actions, loss, state_value_loss}
        self._graph = graph