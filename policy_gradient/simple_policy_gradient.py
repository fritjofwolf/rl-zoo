import tensorflow as tf
import gym
import numpy as np


def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

# for training policy
def train_one_epoch(sess, batch_size):
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

    # render first episode of each epoch
    finished_rendering_this_epoch = False

    # collect experience by acting in the environment with current policy
    while True:
        # save obs
        batch_obs.append(obs.copy())
        # act in the environment
        act = sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]
        obs, rew, done, _ = env.step(act)
        # save action, reward
        batch_acts.append(act)
        ep_rews.append(rew)
        if done:
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            # the weight for each logprob(a_t|s_t) is reward-to-go from t
            batch_weights += list(reward_to_go(ep_rews))
            # reset episode-specific variables
            obs, done, ep_rews = env.reset(), False, []
            # won't render again this epoch
            finished_rendering_this_epoch = True
            # end experience loop if we have enough of it
            if len(batch_rets) > batch_size:
                break
    return batch_obs, batch_acts, batch_weights, batch_rets, batch_lens

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    obs_dim = 4
    n_acts = 2

    # make core of policy network
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32)
    mlp = tf.keras.layers.Dense(n_acts)
    logits = mlp(obs_ph)

    # make action selection op (outputs int actions, sampled from policy)
    actions = tf.squeeze(tf.multinomial(logits=logits,num_samples=1), axis=1)

    # make loss function whose gradient, for the right data, is policy gradient
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32)
    weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32)
    action_masks = tf.one_hot(act_ph, n_acts)
    log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
    loss = -tf.reduce_mean(weights_ph * log_probs)


    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    for i in range(100):
        tmp1, tmp2, tmp3, batch_rets, batch_len = train_one_epoch(sess, 20)
        print(np.mean(batch_len), np.max(batch_len))
        train = optimizer.minimize(loss, var_list=[mlp.kernel, mlp.bias])
        sess.run(train,feed_dict={
                                        obs_ph: np.array(tmp1),
                                        act_ph: np.array(tmp2),
                                        weights_ph: np.array(tmp3)
                                        })