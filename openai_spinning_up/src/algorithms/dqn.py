import tensorflow as tf
import gym
import numpy as np
import logging
import tqdm

class DQN():

    def __init__(self, env_name, model_type = 'dense', gamma = 0.99, learning_rate = 0.00025, replay_buffer_size = 10000,\
                    batch_size = 32):
        self._env = gym.make(env_name)
        self._evaluation_env = gym.make(env_name)
        self._gamma = gamma
        self._batch_size = batch_size
        self._model_type = model_type
        self._learning_rate = learning_rate
        self._replay_buffer_size = replay_buffer_size
        self._obs_dim = self._env.observation_space.shape[0]
        self._n_acts = self._env.action_space.n
        self._build_computational_graph_categorical_actions()
        self._finished_episodes = 0
        self._finished_steps = 0
        self._evaluation_scores = []
        self._row_pointer = 0
        self._exploration_rate = 1
        self._exploration_decay = 0.99999

    def train(self, n_steps = 10**6):
        self._fill_replay_buffer()
        continue_training = True
        for j in tqdm.tqdm_notebook(range(n_steps)):
            state, action, reward, new_state, done = self._sample_experience()
            self._add_sample_to_replay_buffer(state, action, reward, new_state, done)
            self._update_q_function()
            do_evaluation, do_update_target = self._compute_flags()
            if do_update_target:
                self._update_target_network()
            if do_evaluation:
                self._evaluate_model(1)
        self._evaluate_model(100)
        return self._evaluation_scores


    def _compute_flags(self):
        do_evaluation = False
        do_update_target = False
        if self._finished_steps % 1000 == 0:
            do_evaluation = True
        if self._finished_steps % 10000 == 0:
            do_update_target = True
        return do_evaluation, do_update_target


    def _sample_experience(self):
        state = self._state
        self._exploration_rate = max(self._exploration_decay*self._exploration_rate, 0.1)
        action = self._select_action_eps_greedy(state, self._exploration_rate)
        new_state, reward, done, info = self._env.step(action)
        self._finished_steps += 1
        if done:
            self._state = self._env.reset()
            self._finished_episodes += 1
        else:
            self._state = new_state
        return state, action, reward, new_state, done


    def _fill_replay_buffer(self):
        cnt = 0
        while cnt < self._experience_replay_buffer.shape[0]:
            done = False
            state = self._env.reset()
            while not done:
                action = np.random.randint(self._env.action_space.n)
                new_state, reward, done, _ = self._env.step(action)
                self._add_sample_to_replay_buffer(state, action, reward, new_state, done)
                state = new_state
                cnt += 1
        self._state = self._env.reset()


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
        old_state_action_values = self._graph[6]
        obs_ph = self._graph[0]
        sum_return = 0
        sum_q_function = 0
        env = self._evaluation_env
        for i in range(iterations):
            done = False
            state = env.reset()
            while not done:
                action = self._select_action_eps_greedy(state, 0)
                sum_q_function += self._sess.run(old_state_action_values, feed_dict={
                                obs_ph:np.array(state).reshape(-1, self._obs_dim)}
                                )[0][action]
                new_state, reward, done, _ = env.step(action)
                state = new_state
                sum_return += reward
        self._evaluation_scores.append((self._finished_steps, sum_return / iterations, sum_q_function / iterations))
                

    def _update_q_function(self):
        experience_batch = self._sample_experience_batch()
        states, actions, rewards, new_states, terminal_flags = self._extract_data(experience_batch)
        loss = self._gradient_step(states, actions, rewards, new_states, terminal_flags)
        return loss

        
    def _gradient_step(self, states, actions, rewards, new_states, terminal_flags):
        train_action_value = self._graph[7]
        action_value_loss = self._graph[8]
        [obs_ph, act_ph, rew_ph, new_obs_ph, terminal_ph] = self._graph[:5]
        # x = self._graph[-2]
        # y = self._graph[-1]
        # print(np.array(self._sess.run([y], feed_dict={
        #                             obs_ph: np.array(states).reshape(-1, self._obs_dim),
        #                             act_ph: np.array(actions),
        #                             rew_ph: np.array(rewards),
        #                             new_obs_ph: np.array(new_states).reshape(-1, self._obs_dim),
        #                             terminal_ph: np.array(terminal_flags)
        #                         })).shape)
        _, loss = self._sess.run([train_action_value, action_value_loss], feed_dict={
                                    obs_ph: np.array(states).reshape(-1, self._obs_dim),
                                    act_ph: np.array(actions),
                                    rew_ph: np.array(rewards),
                                    new_obs_ph: np.array(new_states).reshape(-1, self._obs_dim),
                                    terminal_ph: np.array(terminal_flags)
                                })
        return loss
        
    def _extract_data(self, experience_batch):
        states = [x[:self._obs_dim] for x in experience_batch]
        actions = [x[2*self._obs_dim] for x in experience_batch]
        rewards = [x[2*self._obs_dim+1] for x in experience_batch]
        new_states = [x[self._obs_dim:2*self._obs_dim] for x in experience_batch]
        terminal_flags = [x[2*self._obs_dim+2] for x in experience_batch]
        return states, actions, rewards, new_states, terminal_flags
        
    def _sample_experience_batch(self):
        indices = np.random.choice(np.arange(self._replay_buffer_size), self._batch_size)
        sample_batch = self._experience_replay_buffer[indices]
        return sample_batch

    def _select_action_eps_greedy(self, state, eps):
        greedy_action = self._graph[5]
        obs_ph = self._graph[0]
        sess = self._sess
        if np.random.rand() < eps:
            action = np.random.randint(self._n_acts)
        else:
            action = sess.run(greedy_action, {obs_ph: state.reshape(1,-1)})[0]
        return action


    def _preprocess_atari_state(self, state):
        pass


    def _build_computational_graph_categorical_actions(self):
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
        if self._model_type == 'dense':
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
                 old_state_action_values, train_action_value, action_value_loss, q_value_network, target_network, old_selected_action_values, y]


    def _build_dense_model(self):
        q_value_network = tf.keras.models.Sequential()
        q_value_network.add(tf.keras.layers.Dense(16, activation='relu'))
        q_value_network.add(tf.keras.layers.Dense(16, activation='relu'))
        q_value_network.add(tf.keras.layers.Dense(self._n_acts, activation='linear'))
        return q_value_network

    # def _build_cnn_model(self):
    #     q_value_network = tf.keras.models.Sequential()
    #     # todo add cnn

    #     return q_value_network