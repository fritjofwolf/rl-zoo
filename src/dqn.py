import tensorflow as tf
import gym
import numpy as np
from src.data_collection import collect_data
import numpy as np
import logging

class DQN():

    def __init__(self, env_name, model_type = 'dense', gamma = 0.99, learning_rate = 0.001, replay_buffer_size = 1000,\
                    batch_size = 32):
        self._env = gym.make(env_name)
        self._evaluation_env = gym.make(env_name)
        self._gamma = gamma
        self._batch_size = batch_size
        self._model_type = model_type
        self._learning_rate = learning_rate
        self._replay_buffer_size = replay_buffer_size
        self._obs_dim = env.observation_space.shape[0]
        self._n_acts = env.action_space.n
        self._build_computational_graph_categorical_actions()

    # def train(self, n_episodes):
    #     global mlp_action_val, mlp_target, new_obs_ph, new_state_action_values
    #     row_pointer = 0
    #     learnable = False
    #     for i in range(n_episodes):
    #         if i % 100 == 0:
    #         done = False
    #         state = env.reset()
    #         cnt = 0
    #         while not done:
    #             action = select_action(sess, state, 0.1)
    #             new_state, reward, done, _ = env.step(action)
    #             add_sample(experience_replay_buffer, state, action, reward, new_state, done, row_pointer)
    #             row_pointer += 1
    #             row_pointer %= exp_replay_size
    #             state = new_state
    #             if learnable:
    #                 update_q_function()
    #             cnt += 1


    def _train(self, n_episodes):
        self._fill_replay_buffer()
        finished_training = False
        while not finished_training:
            state, action, reward, new_state, done = self._sample_experience()
            self._add_sample(state, action, reward, new_state, done)
            self._update_q_function()
            # todo: fix name
            finished_training, evaluate, target_update = self._compute_flags()
            if target_update:
                self._update_target_network()
            if evaluate:
                self._evaluate()


    def _sample_experience(self):
        #todo
        return state, action, reward, new_state, done


    def _fill_replay_buffer(self):
        cnt = 0
        while cnt < self._experience_replay_buffer.shape[0]:
            done = False
            state = env.reset()
            while not done:
                action = np.random.randint(self._env.action_space.n)
                new_state, reward, done, _ = env.step(action)
                self._add_sample_to_replay_buffer(state, action, reward, new_state, done)
                state = new_state
                cnt += 1


    def _update_target_network(self):
        q_value_network = self._graph[9]
        target_network = self._graph[10]
        self._sess.run([v_t.assign(v) for v_t, v in zip(target_network.trainable_weights, q_value_network.trainable_weights)])


    def _add_sample_to_replay_buffer(self, state, action, reward, new_state, done):
        self._experience_replay_buffer[self._row_pointer, :self._obs_dim] = state
        self._experience_replay_buffer[self._row_pointer, self._obs_dim:2*self._obs_dim] = new_state
        self._experience_replay_buffer[self._row_pointer, 2*self._obs_dim] = action
        self._experience_replay_buffer[self._row_pointer, 2*self._obs_dim+1] = reward
        self._experience_replay_buffer[self._row_pointer, 2*self._obs_dim+2] = float(done)
        self._row_pointer += 1
        self._row_pointer %= self._replay_buffer_size
        

    def _evaluate_model(self, iterations):
        sum_return = 0
        env = self._evaluation_env
        for i in range(iterations):
            done = False
            state = env.reset()
            while not done:
                action = self._select_action_eps_greedy(state, 0)
                new_state, reward, done, _ = env.step(action)
                state = new_state
                sum_return += reward
        return sum_return / iterations
                

            
    def _update_q_function():
        experience_batch = self._sample_experience_batch()
        states, actions, rewards, new_states, terminal_flags = self._extract_data(experience_batch)
        loss = self._gradient_step(states, actions, rewards, new_states, terminal_flags)
        return loss

        
    def _gradient_step(self, states, actions, rewards, new_states, terminal_flags):
        _, loss, debug_output = sess.run([train_action_value, action_value_loss, y], feed_dict={
                                    obs_ph: np.array(states).reshape(-1, obs_dim),
                                    act_ph: np.array(actions),
                                    rew_ph: np.array(rewards),
                                    new_obs_ph: np.array(new_states).reshape(-1,obs_dim),
                                    terminal_ph: np.array(terminal_flags)
                                })
        return loss
        
    def _extract_data(self, experience_batch):
        states = [x[:obs_dim] for x in experience_batch]
        actions = [x[2*obs_dim] for x in experience_batch]
        rewards = [x[2*obs_dim+1] for x in experience_batch]
        new_states = [x[obs_dim:2*obs_dim] for x in experience_batch]
        terminal_flags = [x[2*obs_dim+2] for x in experience_batch]
        return states, actions, rewards, new_states, terminal_flags
        
    def _sample_experience_batch(self):
        indices = np.random.choice(np.arange(self._replay_buffer_size), self._batch_size)
        sample_batch = self._experience_replay_buffer[indices]
        return sample_batch

    def _select_action_eps_greedy(self, state, eps):
        greedy_action = self._graph[5]
        obs_ph = self._graph[0]
        if np.random.rand() < eps:
            action = np.random.randint(n_acts)
        else:
            action = sess.run(greedy_action, {obs_ph: state.reshape(1,-1)})[0]
        return action


    def _preprocess_atari_state(self, state):
        pass


    def _build_computational_graph_categorical_actions(self, network_type):
        env = self._env
        obs_dim = self._obs_dim
        n_acts = self._n_acts

        memory_width = 3 + 2*obs_dim
        self._experience_replay_buffer = np.zeros((self._replay_buffer_size, memory_width))

        obs_ph = tf.placeholder(shape=(None,obs_dim), dtype=tf.float32)
        act_ph = tf.placeholder(shape=(None), dtype=tf.int32)
        rew_ph = tf.placeholder(shape=(None), dtype=tf.float32)
        new_obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
        terminal_ph = tf.placeholder(shape=(None), dtype=tf.float32)

        # make core of state-action-value function network
        if network_type == 'dense':
            q_value_network = self._build_dense_model()
            target_network = self._build_dense_model()
        else:
            q_value_network = self._build_cnn_model()
            target_network = self._build_cnn_model()

        # define state action values
        old_state_action_values = q_value_network(obs_ph)
        new_state_action_values = target_network(new_obs_ph)

        # select action
        greedy_action = tf.math.argmax(old_state_action_values, axis=1)

        # define loss function
        y = rew_ph + self._gamma * tf.reduce_max(new_state_action_values, axis=1)*(1-terminal_ph)
        action_masks = tf.one_hot(act_ph, n_acts)
        old_selected_action_values = tf.reduce_sum(action_masks * old_state_action_values, axis=1)
        action_value_loss = tf.losses.mean_squared_error(tf.stop_gradient(y), old_selected_action_values)

        # define optimizer
        optimizer_action_value = tf.train.AdamOptimizer(self._learning_rate)
        train_action_value = optimizer_action_value.minimize(action_value_loss)

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        self._graph = [obs_ph, act_ph, rew_ph, new_obs_ph, terminal_ph, greedy_action,\
                 state_values, train_action_value, action_value_loss, q_value_network, target_network]


    def _build_dense_model(self):
        q_value_network = tf.keras.models.Sequential()
        q_value_network.add(tf.keras.layers.Dense(50, activation='relu'))
        q_value_network.add(tf.keras.layers.Dense(50, activation='relu'))
        q_value_network.add(tf.keras.layers.Dense(n_acts, activation=None))
        return q_value_network

    def _build_cnn_model(self):
        q_value_network = tf.keras.models.Sequential()
        # todo add cnn

        return q_value_network