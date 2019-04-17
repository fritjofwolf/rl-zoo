import tensorflow as tf
import tensorflow_probability as tfp
import keras
import gym
import json
import numpy as np
import logging
from tqdm import tqdm_notebook
from tensorflow.keras.models import load_model
from keras.models import model_from_json

from ..data_collection.a2c_data_collection import A2CDataCollector

class PPO():

    def __init__(self, env_name, gamma = 0.9, learning_rate = 0.0003, model_path = None):
        self._env = gym.make(env_name)
        self._learning_rate = learning_rate
        self._env_name = env_name
        self._gamma = gamma
        self._model_path = model_path
        # self._obs_dim = 100800
        self._obs_dim = self._env.observation_space.shape[0]
        if type(self._env.action_space) == gym.spaces.box.Box:
            self._n_acts = self._env.action_space.shape[0]
            self._action_resizing_factor = self._n_acts
            self._build_computational_graph_continuous_actions()
        else:
            self._n_acts = self._env.action_space.n
            self._action_resizing_factor = 1
            self._build_computational_graph_categorical_actions()


    def train(self, n_epochs, K = 5):
        [obs_ph, act_ph, new_obs_ph, rew_ph, terminal_ph, policy_network, old_policy_network, state_value_network, actions, train_policy, train_state_value] = self._graph
        data_collector = A2CDataCollector(self._sess, self._env_name, actions, obs_ph, 20, 20)
        for i in tqdm_notebook(range(n_epochs)):
            self._update_old_network()
            obs, acts, new_obs, rews, terminal = data_collector.collect_data()
            for j in range(K):
                self._sess.run([train_policy],feed_dict={
                                            obs_ph: np.array(obs).reshape(-1, self._obs_dim),
                                            act_ph: np.array(acts).reshape(-1),
                                            new_obs_ph: np.array(new_obs).reshape(-1, self._obs_dim),
                                            rew_ph: np.array(rews).reshape(-1, 1),
                                            terminal_ph: np.array(terminal).reshape(-1, 1)
                                        })
            for j in range(30):
                self._sess.run([train_state_value],feed_dict={
                                            obs_ph: np.array(obs).reshape(-1, self._obs_dim),
                                            act_ph: np.array(acts).reshape(-1),
                                            new_obs_ph: np.array(new_obs).reshape(-1, self._obs_dim),
                                            rew_ph: np.array(rews).reshape(-1, 1),
                                            terminal_ph: np.array(terminal).reshape(-1, 1)
                                        })
        self._save_model()
        return data_collector.get_episode_statistics()

    def _save_model(self):
        checkpoint_path = '/home/janus/models/'
        # # serialize model to JSON
        # model_json = self._graph[5].to_json()
        # with open(checkpoint_path + "policy_model.json", "w") as json_file:
        #     json_file.write(model_json)
        # # serialize weights to HDF5
        # self._graph[5].save_weights(checkpoint_path+'policy_network.h5')
        self._graph[5].save(checkpoint_path+'policy_network.h5')
        self._graph[7].save(checkpoint_path+'state_network.h5')
        print("Weights after training")
        print(self._graph[5].get_weights())

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
        td_target = rew_ph + self._gamma * new_state_value * (1-terminal_ph)

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
        policy_loss = -tf.reduce_mean(tf.math.minimum(advantages*r, advantages*factor))

        state_value_loss = tf.losses.mean_squared_error(tf.stop_gradient(td_target), state_value)

        policy_optimizer = tf.train.AdamOptimizer(self._learning_rate)
        state_value_optimizer = tf.train.AdamOptimizer(self._learning_rate)
        train_policy = policy_optimizer.minimize(policy_loss)
        train_state_value = state_value_optimizer.minimize(state_value_loss)
        
        if self._model_path:
            # load json and create model
            # json_file = open(self._model_path+'policy_model.json', 'r')
            # loaded_model_json = json_file.read()
            # json_file.close()
            # policy_network = model_from_json(loaded_model_json)
            # # load weights into new model
            # policy_network.load_weights(self._model_path+'policy_network.h5')
            # print("Loaded model from disk")
            # new_model = keras.models.load_model('my_model.h5')
            # new_model.summary()
            policy_network = load_model(self._model_path+'/policy_network.h5')
            old_policy_network = load_model(self._model_path+'/policy_network.h5')
            state_value_network = load_model(self._model_path+'/state_network.h5')
        #     print('Model loaded successfully')
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self._graph = [obs_ph, act_ph, new_obs_ph, rew_ph, terminal_ph, \
                        policy_network, old_policy_network, state_value_network, actions, train_policy, train_state_value]

    def _build_computational_graph_continuous_actions(self):
        # define placeholder
        obs_ph = tf.placeholder(shape=(None, self._obs_dim), dtype=tf.float32)
        act_ph = tf.placeholder(shape=(None,self._n_acts), dtype=tf.float32)
        new_obs_ph = tf.placeholder(shape=(None, self._obs_dim), dtype=tf.float32)
        rew_ph = tf.placeholder(shape=(None,1), dtype=tf.float32)
        terminal_ph = tf.placeholder(shape=(None,1), dtype=tf.float32)

        # build networks
        policy_network = self._build_network('tanh', self._n_acts)
        old_policy_network = self._build_network('tanh', self._n_acts)
        state_value_network = self._build_network('relu', 1)
        
        state_value = state_value_network(obs_ph)
        new_state_value = state_value_network(new_obs_ph)
        td_target = rew_ph + self._gamma * new_state_value * (1-terminal_ph)

        log_std = tf.Variable(-0.5)
        std = tf.math.exp(log_std)

        # make loss function whose gradient, for the right data, is policy gradient
        obs_logits = policy_network(obs_ph)
        obs_logits_old_network = policy_network(obs_ph)
        tfd = tfp.distributions
        dist = tfd.Normal(loc=obs_logits, scale=np.ones(self._n_acts)*std)
        actions = dist.sample(1)[0]

        Z = (2*np.pi*std**2)**0.5
        selected_action_probs = tf.math.exp(-0.5*(act_ph - obs_logits)**2 / std**2) / Z
        selected_action_probs_old_network = tf.math.exp(-0.5*(act_ph - obs_logits_old_network)**2 / std**2) / Z

        # action_masks = tf.one_hot(act_ph, self._n_acts)
        # selected_action_probs = tf.reduce_sum(action_masks * tf.nn.softmax(obs_logits), axis=1)
        # selected_action_probs_old_network = tf.reduce_sum(action_masks * tf.nn.softmax(obs_logits_old_network), axis=1)

        r = selected_action_probs / tf.stop_gradient(selected_action_probs_old_network)
        advantages = td_target - state_value
        factor = 1 + 0.2 * tf.math.sign(advantages)
        x = tf.math.minimum(advantages*r, advantages*factor)
        policy_loss = -tf.reduce_mean(x)

        state_value_loss = tf.losses.mean_squared_error(tf.stop_gradient(td_target), state_value)

        policy_optimizer = tf.train.AdamOptimizer(self._learning_rate)
        state_value_optimizer = tf.train.AdamOptimizer(self._learning_rate)
        train_policy = policy_optimizer.minimize(policy_loss)
        train_state_value = state_value_optimizer.minimize(state_value_loss)
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        self._graph = [obs_ph, act_ph, new_obs_ph, rew_ph, terminal_ph, \
                        policy_network, old_policy_network, actions, train_policy, train_state_value]


    def _build_network(self, activation = 'relu', n_output_units = 1):
        mlp = keras.models.Sequential()
        mlp.add(keras.layers.Dense(16, activation=activation, input_shape=(self._obs_dim,)))
        mlp.add(keras.layers.Dense(16, activation=activation))
        mlp.add(keras.layers.Dense(n_output_units, activation=None))
        return mlp