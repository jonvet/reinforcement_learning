
# conda install libgcc

import gym
import pandas as pd
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from skimage import color
from matplotlib import pyplot as plt
import os
import pickle as pkl

class RandomAgent(object):

    def __init__(self, game):
        self.game = game
        self.env = gym.make(self.game)
        self.env._max_episode_steps = 3000
        self.env.action_space = self.env.action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class AgentQ(object):
    
    def __init__(self, game, eta, optimizer, discount = 0.99, num_epochs = 10, batch_size = 50, keep_prob_dropout = 1.0, L2 = 0, 
        num_fixed = 5000, clip = None, epsilon = 0.05, epsilon_decay = 1.0, decay = 1.0, decay_step = 10000, new_episodes = 10, buffer_size = 100000, train = True, save_every = 500, load_path = None):

        self.game = game
        self.eta = eta
        self.optimizer = optimizer
        self.discount = discount
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.keep_prob_dropout = keep_prob_dropout
        self.L2 = L2
        self.num_fixed = num_fixed
        self.clip = clip
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.decay = decay
        self.decay_step = decay_step
        self.new_episodes = new_episodes
        self.buffer_size = buffer_size
        self.train = train
        self.losses = np.asarray([np.inf])
        self.lengths = np.array([0])
        self.returns = np.array([-1])
        self.save_every = save_every
        self.load_path = load_path

        self.env = gym.make(self.game)
        self.env._max_episode_steps = 3000
        self.num_actions = self.env.action_space.n
        self.env.close()

        self.current_l1_weight = np.random.normal(0, 0.2, [6,6,4,16])
        self.current_l2_weight = np.random.normal(0, 0.2, [4,4,16,32])
        self.current_l3_weight = np.random.normal(0, 0.2, [7*7*32, 256])
        self.current_l3_bias = np.random.normal(0, 0.2, [256])
        self.current_l4_weight = np.random.normal(0, 0.2, [256, self.num_actions])
        self.current_l4_bias = np.random.normal(0, 0.2, [self.num_actions])

    def save_model(self, step):
        saver = tf.train.Saver(write_version = tf.train.SaverDef.V1)
        if not os.path.exists('./models/task_b/'):
            os.mkdir('./models/task_b')
        saver.save(self.sess, './models/task_b/step_%d.checkpoint' % step)

    def act(self, observation, greedy = False):

        '''
        Takes an observation and returns an action
        If greedy = True, returns always the greedy action as per the target Q function
        If greedy = False, returns a random action with epsilon probability, and the greedy action otherwise
        '''

        observation = np.reshape(observation, [-1, 784*4])
        ran = np.random.random()

        if greedy:
            action = self.sess.run(self.action, feed_dict = {self.states_t: observation, self.dropout: 1.0,
            self.l1_weight_old: self.current_l1_weight, self.l2_weight_old: self.current_l2_weight, 
            self.l3_weight_old: self.current_l3_weight, self.l3_bias_old: self.current_l3_bias,
            self.l4_weight_old: self.current_l4_weight, self.l4_bias_old: self.current_l4_bias})
            return action[0]
        else:
            if ran < self.epsilon:
                return np.random.randint(self.num_actions)
            else:
                action = self.sess.run(self.action, feed_dict = {self.states_t: observation, self.dropout: 1.0,
                self.l1_weight_old: self.current_l1_weight, self.l2_weight_old: self.current_l2_weight, 
                self.l3_weight_old: self.current_l3_weight, self.l3_bias_old: self.current_l3_bias,
                self.l4_weight_old: self.current_l4_weight, self.l4_bias_old: self.current_l4_bias})
                return action[0]

    def get_L2_loss(self):
        all_vars = tf.trainable_variables() 
        return tf.add_n([ tf.nn.l2_loss(v) for v in all_vars if 'bias' not in v.name ]) * self.L2 

    def Q_function(self, s, fixed):

        '''
        Takes an observation and returns a vector of Q-values
        if fixed = True, uses the (fixed) target Q function
        '''

        s = tf.reshape(s, [-1, 28, 28, 4])
        if fixed == True:
            conv1 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(s, self.l1_weight_old, strides=[1, 2, 2, 1], padding='SAME')), keep_prob = self.keep_prob_dropout)
            conv2 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(conv1, self.l2_weight_old, strides=[1, 2, 2, 1], padding='SAME')), keep_prob = self.keep_prob_dropout)
            conv2_flat = tf.reshape(conv2, [-1, 7*7*32])
            fc1 = tf.nn.relu(tf.matmul(conv2_flat, self.l3_weight_old) + self.l3_bias_old)
            q = tf.matmul(fc1, self.l4_weight_old) + self.l4_bias_old     
        else:
            conv1 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(s, self.l1_weight, strides=[1, 2, 2, 1], padding='SAME')), keep_prob = self.keep_prob_dropout)
            conv2 = tf.nn.dropout(tf.nn.relu(tf.nn.conv2d(conv1, self.l2_weight, strides=[1, 2, 2, 1], padding='SAME')), keep_prob = self.keep_prob_dropout)
            conv2_flat = tf.reshape(conv2, [-1, 7*7*32])
            fc1 = tf.nn.relu(tf.matmul(conv2_flat, self.l3_weight) + self.l3_bias)
            q = tf.matmul(fc1, self.l4_weight) + self.l4_bias
        return q

    def play(self, num_episodes, max_episode_steps, random = False, discount = 0.99, render = False, greedy = False):

        '''
        Plays num_episodes with max length of max_episode_steps
        Returns a list of numpy arrays containing S_t, A_t, R_t and S_t+1
        If random = True, plays a random policy
        If greedy = True, takes always the greedy action, otherwise takes a random action with epsilon probability
        '''

        episode_returns = []
        episode_lengths = []
        episode_actions = []
        episode_observations = []
        episode_rewards = []
        transitions = 0
        env = gym.make(self.game)
        env._max_episode_steps = 3000

        for i_episode in range(num_episodes):
            frame = self.preprocess(env.reset())
            observation = np.concatenate((frame, frame, frame, frame), 1)
            reward = 0
            done = False
            episode_return = 0
            step_observations = [observation]
            step_actions = []
            step_rewards = []

            for t in range(max_episode_steps):
                transitions += 1
                if random:
                    action = np.random.randint(self.num_actions)
                else:
                    action = self.act(observation, greedy)
                frame, reward, done, _= env.step(action)
                reward = self.reward_wrapper(reward)
                frame = self.preprocess(frame)
                observation = np.concatenate((observation[:,1:], frame), 1)
                if render:
                    env.render()

                episode_return += reward * self.discount**t 
                step_actions.append(action)
                step_rewards.append(reward)
                step_observations.append(observation)

                if done:
                    break
                
            episode_actions.append(step_actions)
            episode_rewards.append(step_rewards)    
            episode_observations.append(step_observations)
            episode_returns.append(episode_return)
            episode_lengths.append(t+1)

        data_actions = []
        data_rewards = []
        data_observations = []
        data_observations_t = []
        data_lengths = []
        data_returns = []
        for i in range(len(episode_actions)):
            for j in range(len(episode_actions[i])):
                data_actions.append(episode_actions[i][j])
                data_rewards.append(episode_rewards[i][j])
                data_observations.append(episode_observations[i][j])
                data_observations_t.append(episode_observations[i][j+1])
            data_lengths.append(episode_lengths[i])
            data_returns.append(episode_returns[i])
        train_data = [np.asarray(data_observations), np.asarray(data_actions), np.asarray(data_rewards), np.asarray(data_observations_t)]
        test_data = [np.asarray(data_lengths), np.asarray(data_returns)]
        env.close()
        return train_data, test_data

    def rgb2gray(self, rgb):
        return color.rgb2gray(rgb)*255

    def preprocess(self, image):
        gray = self.rgb2gray(image)
        resized = resize(gray, [28, 28])
        flat = np.reshape(resized, [784, 1]).astype(np.uint8)
        return flat

    def reward_wrapper(self, reward):

        '''
        Takes a reward, returns a clipped reward (between -1 and 1)
        '''
        if reward > 1:
            return 1
        elif reward <-1:
            return -1
        else:
            return reward

    def update_buffer(self, current_buffer, new_data, buffer_size):

        '''
        Takes as input the current buffer, new data to be added, and the max size of the buffer.
        Randomly shuffles the current buffer and concatenates the new transitions to it.
        Returns an updated buffer containing the last buffer_size transitions.
        '''

        perm = np.random.permutation(len(current_buffer[0]))
        data_observations = np.concatenate((current_buffer[0][perm], new_data[0]), 0)[-buffer_size:]
        data_actions = np.concatenate((current_buffer[1][perm], new_data[1]), 0)[-buffer_size:]
        data_rewards = np.concatenate((current_buffer[2][perm], new_data[2]), 0)[-buffer_size:]
        data_observations_t = np.concatenate((current_buffer[3][perm], new_data[3]), 0)[-buffer_size:]

        replay_buffer = [data_observations, data_actions, data_rewards, data_observations_t]
        print(len(data_observations))
        return replay_buffer

    def training(self, data):

        '''
        Takes as input a replay buffer and trains the model for self.num_epochs epochs
        Updates the target network every self.num_fixed batches
        Decays the learning rate and epsilon every self.decay_step batches
        Evaluates and saved the model every self.save_every batches
        '''

        for self.epoch in range(self.num_epochs):
            length_data = len(data[0])
            perm = np.random.permutation(length_data)
            data_observations_t = data[0][perm]
            data_actions = data[1][perm]
            data_rewards = data[2][perm]
            data_observations_t_1 = data[3][perm]
            data_observations_t = np.reshape(data_observations_t, [-1, 784 * 4])
            data_observations_t_1 = np.reshape(data_observations_t_1, [-1, 784 * 4])
            data_final_step = np.ones_like(data_rewards)
            data_final_step[data_rewards == -1] = 0
            sum_loss = 0

            for batch in range(length_data // self.batch_size):
                batch_states_t = data_observations_t[batch * self.batch_size: (batch + 1) * self.batch_size, :]
                batch_states_t_1 =data_observations_t_1[batch * self.batch_size: (batch + 1) * self.batch_size]
                batch_actions = data_actions[batch * self.batch_size: (batch + 1) * self.batch_size]
                batch_rewards = data_rewards[batch * self.batch_size: (batch + 1) * self.batch_size]
                batch_final_step = data_final_step[batch * self.batch_size: (batch + 1) * self.batch_size]
                feed_dict = {self.states_t: batch_states_t, self.states_t_1: batch_states_t_1, self.actions: batch_actions, 
                self.rewards: batch_rewards, self.dropout: self.keep_prob_dropout, self.final_step: batch_final_step,
                self.l1_weight_old: self.current_l1_weight, self.l2_weight_old: self.current_l2_weight, 
                self.l3_weight_old: self.current_l3_weight, self.l3_bias_old: self.current_l3_bias,
                self.l4_weight_old: self.current_l4_weight, self.l4_bias_old: self.current_l4_bias}

                _, this_loss, actions, w1, w2, w3, b3, w4, b4 = self.sess.run([self.opt_op, self.loss, self.action, 
                    self.l1_weight, self.l2_weight, self.l3_weight, self.l3_bias, self.l4_weight, self.l4_bias], feed_dict = feed_dict)
                
                sum_loss += this_loss
                step = self.global_step.eval(session = self.sess)
                print('\rStep: %d, loss: %f' % (step, this_loss), end='')

                if (step % self.num_fixed == 0) and ((batch > 0) or (length_data == 1)):
                    self.current_l1_weight = w1
                    self.current_l2_weight = w2
                    self.current_l3_weight = w3
                    self.current_l3_bias = b3
                    self.current_l4_weight = w4
                    self.current_l4_bias = b4

                if (step % self.decay_step == 0):
                    self.epsilon = self.epsilon * self.epsilon_decay
                    print('New epsilon:', self.epsilon)

                if (step % self.save_every == 0):

                    avg_loss = sum_loss/self.save_every
                    sum_loss = 0

                    _, eval_data = self.play(10, 5000, random= False, render = False, greedy = True)

                    print('Step %d, mean length: %f, mean return: %f, mean loss: %f' %(step, np.mean(eval_data[0]), np.mean(eval_data[1]), avg_loss))
                    if np.mean(eval_data[1]) >= np.max(self.returns):
                        self.save_model(step)

                    self.lengths = np.concatenate((self.lengths, [np.mean(eval_data[0])]), 0)
                    self.returns = np.concatenate((self.returns, [np.mean(eval_data[1])]), 0)
                    self.losses = np.concatenate((self.losses, np.array([avg_loss])),0)

                    _, eval_data = self.play(1, 5000, random= False, render = False)

            # fig = plt.subplots(nrows=3, ncols=1)
            # plt.tight_layout()
            # p = plt.subplot(3, 1, 1)
            # p.set_title("Average discounted episode return", fontsize = 8)
            # p.plot(returns[1:])
            # p.set_ylim(-1.5, 0)
            # p = plt.subplot(3, 1, 2)
            # p.set_title("Average episode length", fontsize = 8)
            # p.plot(lengths[1:])
            # p.set_ylim(1000, 3000)
            # p = plt.subplot(3, 1, 3)
            # p.set_title("Average loss over 50 batches", fontsize = 8)
            # p.plot(losses[1:])
            # if not os.path.exists('./plots/task_b/'):
            #     os.mkdir('./plots/task_b')
            # plt.savefig('./plots/task_b/this.png')
            # plt.close()

            dump_this = [self.lengths, self.returns, self.losses]
            with open('dump_this.pkl', 'wb') as f:
                pkl.dump(dump_this, f)
                

    def run_random(self):

        '''
        Function to evaluate a random policy for part B1
        '''

        _, self.evaluation_data = self.play(100, 3000, random = True, render = False)
        lengths, returns = self.evaluation_data
        print(pd.DataFrame(lengths).describe())
        print(pd.DataFrame(returns).describe())

    def run_no_train(self):

        '''
        Function to evaluate an untrained Q agent for part B2
        '''

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.states_t = tf.placeholder(tf.float32, [None, 784 * 4], 'observation_t')
            self.states_t_1 = tf.placeholder(tf.float32, [None, 784 * 4], 'observation_t_1')
            self.actions = tf.placeholder(tf.int32, [None], 'actions')
            self.rewards = tf.placeholder(tf.float32, [None], 'rewards')
            self.dropout = tf.placeholder(tf.float32, name='dropout')
            self.final_step = tf.placeholder(tf.float32, [None], name='final_step')
            self.l1_weight_old = tf.placeholder(tf.float32, [6,6,4,16])
            self.l1_weight = tf.Variable(tf.random_normal([6,6,4,16], mean = 0.0, stddev = 0.2))
            self.l2_weight_old = tf.placeholder(tf.float32, [4,4,16,32])
            self.l2_weight = tf.Variable(tf.random_normal([4,4,16,32], mean = 0.0, stddev = 0.2))
            self.l3_weight_old = tf.placeholder(tf.float32, [7*7*32, 256])
            self.l3_bias_old = tf.placeholder(tf.float32, [256])
            self.l3_weight = tf.Variable(tf.random_normal([7*7*32, 256], mean = 0.0, stddev = 0.2))
            self.l3_bias = tf.Variable(tf.random_normal([256], mean = 0.0, stddev = 0.2))
            self.l4_weight_old = tf.placeholder(tf.float32, [256, self.num_actions])
            self.l4_bias_old = tf.placeholder(tf.float32, [self.num_actions])
            self.l4_weight = tf.Variable(tf.random_normal([256, self.num_actions], mean = 0.0, stddev = 0.5))
            self.l4_bias = tf.Variable(tf.random_normal([self.num_actions], mean = 0.0, stddev = 0.5))
            self.global_step = tf.Variable(0, name = 'global_step', trainable = False)

            with tf.variable_scope("training") as varscope:
                Q_target = tf.reduce_max(self.Q_function(self.states_t_1, fixed = True), 1)
                target = self.rewards + self.discount * Q_target * self.final_step
                self.actions_one_hot = tf.one_hot(self.actions, self.num_actions)
                Q_actual = self.Q_function(self.states_t, fixed = False)
                actual = tf.reduce_sum(tf.multiply(Q_actual, self.actions_one_hot),1)
                self.loss = tf.reduce_mean(tf.square(target - actual)/2) + self.get_L2_loss()
                learning_rate = tf.train.exponential_decay(self.eta, self.global_step, self.decay_step, self.decay, staircase=True)
                self.opt_op = tf.contrib.layers.optimize_loss(loss = self.loss, global_step = self.global_step, learning_rate = learning_rate, optimizer = self.optimizer, clip_gradients= self.clip, 
                    learning_rate_decay_fn=None, summaries = ['loss']) 

            with tf.variable_scope("control") as varscope:
                q_values = self.Q_function(self.states_t, fixed = False)
                self.action = tf.argmax(q_values, 1)

            with tf.Session(graph = self.graph) as self.sess:
                tf.global_variables_initializer().run()
                _, self.evaluation_data = self.play(100, 3000, random = False, render = False, greedy = True)
                lengths, returns = self.evaluation_data
                print(pd.DataFrame(lengths).describe())
                print(pd.DataFrame(returns).describe())


    def run(self):

        '''
        Main function used to train a Q agent with the given parameters, or to test a saved model
        '''

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.states_t = tf.placeholder(tf.float32, [None, 784 * 4], 'observation_t')
            self.states_t_1 = tf.placeholder(tf.float32, [None, 784 * 4], 'observation_t_1')
            self.actions = tf.placeholder(tf.int32, [None], 'actions')
            self.rewards = tf.placeholder(tf.float32, [None], 'rewards')
            self.dropout = tf.placeholder(tf.float32, name='dropout')
            self.final_step = tf.placeholder(tf.float32, [None], name='final_step')
            self.l1_weight_old = tf.placeholder(tf.float32, [6,6,4,16])
            self.l1_weight = tf.Variable(tf.random_normal([6,6,4,16], mean = 0.0, stddev = 0.2))
            self.l2_weight_old = tf.placeholder(tf.float32, [4,4,16,32])
            self.l2_weight = tf.Variable(tf.random_normal([4,4,16,32], mean = 0.0, stddev = 0.2))
            self.l3_weight_old = tf.placeholder(tf.float32, [7*7*32, 256])
            self.l3_bias_old = tf.placeholder(tf.float32, [256])
            self.l3_weight = tf.Variable(tf.random_normal([7*7*32, 256], mean = 0.0, stddev = 0.2))
            self.l3_bias = tf.Variable(tf.random_normal([256], mean = 0.0, stddev = 0.2))
            self.l4_weight_old = tf.placeholder(tf.float32, [256, self.num_actions])
            self.l4_bias_old = tf.placeholder(tf.float32, [self.num_actions])
            self.l4_weight = tf.Variable(tf.random_normal([256, self.num_actions], mean = 0.0, stddev = 0.5))
            self.l4_bias = tf.Variable(tf.random_normal([self.num_actions], mean = 0.0, stddev = 0.5))
            self.global_step = tf.Variable(0, name = 'global_step', trainable = False)

            with tf.variable_scope("training") as varscope:
                Q_target = tf.reduce_max(self.Q_function(self.states_t_1, fixed = True), 1)
                target = self.rewards + self.discount * Q_target * self.final_step
                self.actions_one_hot = tf.one_hot(self.actions, self.num_actions)
                Q_actual = self.Q_function(self.states_t, fixed = False)
                actual = tf.reduce_sum(tf.multiply(Q_actual, self.actions_one_hot),1)
                self.loss = tf.reduce_mean(tf.square(target - actual)/2) + self.get_L2_loss()
                learning_rate = tf.train.exponential_decay(self.eta, self.global_step, self.decay_step, self.decay, staircase=True)
                self.opt_op = tf.contrib.layers.optimize_loss(loss = self.loss, global_step = self.global_step, learning_rate = learning_rate, optimizer = self.optimizer, clip_gradients= self.clip, 
                    learning_rate_decay_fn=None, summaries = ['loss']) 

            with tf.variable_scope("control") as varscope:
                q_values = self.Q_function(self.states_t, fixed = False)
                self.action = tf.argmax(q_values, 1)

        
            if self.train:
                with tf.Session(graph = self.graph) as self.sess:
                    tf.global_variables_initializer().run()
                    print('Playing random to get initial buffer')
                    data, _ = self.play(5, 5000, random= True, render = False)
                    print('Starting training')
                    while self.global_step.eval() < 1000000:
                        self.training(data)
                        more_data, _ = self.play(self.new_episodes, 5000, random = False, render = False)
                        data = self.update_buffer(data, more_data, self.buffer_size)
            else:
                with tf.Session(graph = self.graph) as self.sess:
                    saver = tf.train.Saver()
                    saver.restore(self.sess, self.load_path)
                    print('Model restored')
                    _, self.evaluation_data = self.play(100, 3000, random = False, render = False, greedy = True)
                    lengths, returns = self.evaluation_data
                    print(pd.DataFrame(lengths).describe())
                    print(pd.DataFrame(returns).describe())

# if __name__ == '__main__':

games = ['Pong-v3', 'MsPacman-v3', 'Boxing-v3']

def task_b_1():
    for game in games:
        print('Random policy playing %s' % game)
        agent = AgentQ(game, eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 0.9, L2 = 0, 
        num_fixed = 5000, save_every = 25000,clip = None, epsilon = 0.10, decay = 1.0, train = True)
        agent.run_random()
        print('±±±±±±±±±±±±±±±±±±\n')

def task_b_2():
    for game in games:
        print('Untrained Q playing %s' % game)
        agent = AgentQ(game, eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 0.9, L2 = 0, 
        num_fixed = 5000, save_every = 25000,clip = None, epsilon = 0.10, decay = 1.0, train = True)
        agent.run_no_train()
        print('±±±±±±±±±±±±±±±±±±\n')

# task_b_1()
# task_b_2()

# Pong run 1
# agent = AgentQ(games[0], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, 
    # num_fixed = 5000, save_every = 25000,clip = None, epsilon = 0.10, decay = 1.0, decay_step = 25000, new_episodes = 10, buffer_size = 100000, train = True)

# Pong run 2
# agent = AgentQ(games[0], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, 
#     num_fixed = 5000, save_every = 25000,clip = None, epsilon = 0.10, decay = 0.95, decay_step = 100000, new_episodes = 15, buffer_size = 200000, train = True)

# Pong run 3
# agent = AgentQ(games[0], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 0.9, L2 = 0, 
#     num_fixed = 5000, save_every = 25000,clip = None, epsilon = 0.20, epsilon_decay = 0.9, decay = 0.95, decay_step = 100000, new_episodes = 15, buffer_size = 200000, train = True)

# Pacman run 1
# agent = AgentQ(games[1], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, 
#     num_fixed = 5000, save_every = 25000,clip = None, epsilon = 0.10, decay = 1.0, decay_step = 25000, new_episodes = 10, buffer_size = 100000, train = True)

# Pacman run 2
# agent = AgentQ(games[1], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, 
#     num_fixed = 5000, save_every = 25000,clip = None, epsilon = 0.10, decay = 0.95, decay_step = 100000, new_episodes = 10, buffer_size = 200000,train = True)

# Pacman run 3
# agent = AgentQ(games[1], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, 
#     num_fixed = 500, save_every = 25000,clip = None, epsilon = 0.10, decay = 0.95, decay_step = 100000, new_episodes = 15, buffer_size = 200000,train = True)

# Boxing run 1
# agent = AgentQ(games[2], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, 
#     num_fixed = 5000, save_every = 25000,clip = None, epsilon = 0.10, decay = 1.0, decay_step = 100000, new_episodes = 10, buffer_size = 100000, train = True)

# Boxing run 2
# agent = AgentQ(games[2], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 64, keep_prob_dropout = 0.9, L2 = 0, 
#     num_fixed = 500, save_every = 25000,clip = None, epsilon = 0.10, epsilon_decay = 1.0, decay = 0.95, decay_step = 100000, new_episodes = 5, buffer_size = 200000, train = True)


# Load model
# agent = AgentQ(games[0], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, 
#     num_fixed = 5000, clip = None, epsilon = 0, decay = 1.0, train = False, load_path = '../models/task_b_pong_run_1/step_550000.checkpoint')
# agent = AgentQ(games[0], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, 
#     num_fixed = 5000, clip = None, epsilon = 0, decay = 1.0, train = False, load_path = '../models/task_b_pong_run_2/step_950000.checkpoint')
# agent = AgentQ(games[0], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, 
#     num_fixed = 5000, clip = None, epsilon = 0, decay = 1.0, train = False, load_path = '../models/task_b_pong_run_3/step_650000.checkpoint')

# agent = AgentQ(games[1], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, 
#     num_fixed = 5000, clip = None, epsilon = 0, decay = 1.0, train = False, load_path = '../models/task_b_pacman_run_1/step_150000.checkpoint')
# agent = AgentQ(games[1], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, 
#     num_fixed = 5000, clip = None, epsilon = 0, decay = 1.0, train = False, load_path = '../models/task_b_pacman_run_2/step_400000.checkpoint')
agent = AgentQ(games[1], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, 
    num_fixed = 5000, clip = None, epsilon = 0, decay = 1.0, train = False, load_path = '../models/task_b_pacman_run_3/step_100000.checkpoint')

# agent = AgentQ(games[2], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, 
#     num_fixed = 5000, clip = None, epsilon = 0, decay = 1.0, train = False, load_path = '../models/task_b_boxing_run_1/step_200000.checkpoint')
# agent = AgentQ(games[2], eta = 0.001, optimizer = 'RMSProp', discount = 0.99, num_epochs = 1, batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, 
#     num_fixed = 5000, clip = None, epsilon = 0, decay = 1.0, train = False, load_path = '../models/task_b_boxing_run_2/step_25000.checkpoint')


agent.run()   
