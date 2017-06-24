import gym
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import os
import pickle as pkl

class RandomAgent(object):

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class AgentQ(object):
    
    '''
    Q-Agent class defined by the given parameters:
    - action_space: Defines the actions the agent can take (integers)
    - eta: learning rate
    - discount: rate with which rewards are discounted
    - nn: Boolean. Creates a layer of hidden units if True
    - layers: number of hidden layers, if nn = True
    - units: number of units in the hidden layers, if nn = True
    - relu: If True, adds a relu layer in the hidden layer
    - num_epochs: Number of epochs to train on a data set/ replay buffer
    - batch_size: number of transitions per batch
    - keep_prob_dropout: 1 - probability of dropout in the hidden layer
    - L2: Amount of L2 regularisation
    - num_fixed: Number of batches for which the target network is fixed
    - clip: Amount of gradient clipping
    - epsilon: percentage chance that random action is chosen rather than greedy action
    - decay: percentage learning rate decay
    - decay_steps: number of batches after which learning rate decays
    - double_Q: Boolean, if True, trains a double-Q agent
    - load_path: If None, trains an agent, otherwise, loads a trained model from load_path
    - external_session: Session to be used for loading an agent
    '''

    def __init__(self, action_space, eta, optimizer, discount = 0.99, nn = False, layers = 1, units = 100, relu = True, 
            num_epochs = 1, batch_size = 50, keep_prob_dropout = 1.0, L2 = 0, num_fixed = 5, clip = None, epsilon = 0, 
            decay = 1.0, decay_steps = 10000, double_Q = False, load_path = None, external_session = None):
        self.action_space = action_space
        self.eta = eta
        self.optimizer = optimizer
        self.discount = discount
        self.nn = nn
        self.layers = layers
        self.units = units
        self.relu = relu
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.keep_prob_dropout = keep_prob_dropout
        self.L2 = L2
        self.num_fixed = num_fixed
        self.clip = clip
        self.graph = tf.get_default_graph()
        self.epsilon = epsilon
        self.decay = decay
        self.decay_steps = decay_steps
        self.double_Q = double_Q
        self.losses = np.asarray([np.inf])
        self.lengths = np.array([0])
        self.returns = np.array([-1])
        self.load_path = load_path

        self.current_weight = np.random.normal(0, 0.2, [self.units, 2])
        self.current_bias = np.random.normal(0, 0.2, [2])
        self.current_weight_h = np.random.normal(0, 0.2, [4, self.units])
        self.current_bias_h = np.random.normal(0, 0.2, [self.units])

        if self.double_Q == True:
            self.current_weight_2 = np.random.normal(0, 0.2, [self.units, 2])
            self.current_bias_2 = np.random.normal(0, 0.2, [2])
            self.current_weight_h_2 = np.random.normal(0, 0.2, [4, self.units])
            self.current_bias_h_2 = np.random.normal(0, 0.2, [self.units])

        # with self.graph.as_default():
        self.initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.2)
        self.states_t = tf.placeholder(tf.float32, [None, 4], 'observation_t')
        self.states_t_1 = tf.placeholder(tf.float32, [None, 4], 'observation_t_1')
        self.actions = tf.placeholder(tf.int32, [None], 'actions')
        self.rewards = tf.placeholder(tf.float32, [None], 'rewards')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.Q_draw = tf.placeholder(tf.float32, name='double_Q_draw')
        self.linear_weight_old = tf.placeholder(tf.float32, [self.units, 2], 'old_weight')
        self.linear_bias_old = tf.placeholder(tf.float32, [2], 'old_bias')
        self.linear_weight = tf.get_variable('linear_weight', [self.units, 2], tf.float32, initializer = self.initializer)
        self.linear_bias = tf.get_variable('linear_bias', [2], tf.float32, initializer = self.initializer)

        if self.nn:
            self.hidden_weight_old = tf.placeholder(tf.float32, [4, self.units], 'old_weight_h')
            self.hidden_bias_old = tf.placeholder(tf.float32, [self.units], 'old_bias_h')
            self.hidden_weight = tf.get_variable('hidden_weight', [4, self.units], tf.float32, initializer = self.initializer)
            self.hidden_bias = tf.get_variable('hidden_bias', [self.units], tf.float32, initializer = self.initializer)

        if self.double_Q == True:
            self.linear_weight_2 = tf.get_variable('linear_weight_2', [self.units, 2], tf.float32, initializer = self.initializer)
            self.linear_bias_2 = tf.get_variable('linear_bias_2', [2], tf.float32, initializer = self.initializer)
            self.hidden_weight_2 = tf.get_variable('hidden_weight_2', [4, self.units], tf.float32, initializer = self.initializer)
            self.hidden_bias_2 = tf.get_variable('hidden_bias_2', [self.units], tf.float32, initializer = self.initializer)

        self.global_step = tf.Variable(0, name = 'global_step')

        with tf.variable_scope("training") as varscope:
            self.actions_one_hot = tf.one_hot(self.actions, 2)
            
            if self.double_Q == True:
                with tf.variable_scope("double_Q_loss_1") as varscope: # Update function 1

                    Q_target1 = self.double_Q_function_1(self.states_t_1)
                    target1 = self.rewards + self.discount * Q_target1 * (self.rewards+1)
                    
                    actual_hidden1 = tf.nn.relu(tf.nn.dropout(tf.matmul(self.states_t, self.hidden_weight) + self.hidden_bias, keep_prob = self.dropout))
                    actual_q1 = tf.matmul(actual_hidden1, self.linear_weight) + self.linear_bias
                    actual1 = tf.reduce_sum(tf.multiply(actual_q1, self.actions_one_hot),1)

                    self.loss1 = tf.reduce_mean(tf.square(target1 - actual1)/2) 
                    learning_rate1 = tf.train.exponential_decay(self.eta, self.global_step, self.decay_steps, self.decay, staircase=True)
                    self.opt_op1 = tf.contrib.layers.optimize_loss(loss = self.loss1, global_step = self.global_step, learning_rate = learning_rate1, 
                        optimizer = self.optimizer, clip_gradients= self.clip, learning_rate_decay_fn=None, summaries = ['loss']) 

                with tf.variable_scope("double_Q_loss_2") as varscope: # Update function 2
                    Q_target2 = self.double_Q_function_2(self.states_t_1)
                    target2 = self.rewards + self.discount * Q_target2 * (self.rewards+1)

                    actual_hidden2 = tf.nn.relu(tf.nn.dropout(tf.matmul(self.states_t, self.hidden_weight_2) + self.hidden_bias_2, keep_prob = self.dropout))
                    actual_q2 = tf.matmul(actual_hidden2, self.linear_weight_2) + self.linear_bias_2
                    actual2 = tf.reduce_sum(tf.multiply(actual_q2, self.actions_one_hot),1)

                    self.loss2 = tf.reduce_mean(tf.square(target2 - actual2)/2) 
                    learning_rate2 = tf.train.exponential_decay(self.eta, self.global_step, self.decay_steps, self.decay, staircase=True)
                    self.opt_op2 = tf.contrib.layers.optimize_loss(loss = self.loss2, global_step = self.global_step, learning_rate = learning_rate2, 
                        optimizer = self.optimizer, clip_gradients= self.clip, learning_rate_decay_fn=None, summaries = ['loss']) 

            else:
                with tf.variable_scope("Q_loss") as varscope:
                    Q_target = tf.reduce_max(self.Q_function(self.states_t_1, fixed = True), 1)
                    target = self.rewards + self.discount * Q_target * (self.rewards+1)

                    Q_actual = self.Q_function(self.states_t, fixed = False)
                    actual = tf.reduce_sum(tf.multiply(Q_actual, self.actions_one_hot),1)

                    self.loss = tf.reduce_mean(tf.square(target - actual)/2) 
                    learning_rate = tf.train.exponential_decay(self.eta, self.global_step, self.decay_steps, self.decay, staircase=True)
                    self.opt_op = tf.contrib.layers.optimize_loss(loss = self.loss, global_step = self.global_step, learning_rate = learning_rate, 
                        optimizer = self.optimizer, clip_gradients= self.clip, learning_rate_decay_fn=None, summaries = ['loss']) 

        with tf.variable_scope("control") as varscope:
            q_values = self.Q_function(self.states_t, fixed = True)
            self.action = tf.argmax(q_values, 1)

        if self.load_path == None:
            # self.sess =  tf.Session(graph=tf.get_default_graph())
            self.sess = external_session
            tf.global_variables_initializer().run(session = self.sess)
            print('Graph initialised for training')

        else:
            # self.sess = tf.Session(graph=self.graph)
            self.sess = external_session
            # saver = tf.train.Saver()
            # saver.restore(self.sess, self.load_path)
            
            env = gym.make('CartPole-v0')
            env._max_episode_steps = 300
            env.render()

            with open(self.load_path, 'rb') as f:
                weights = pkl.load(f)

            print('model loaded!')

            this_task2 = task(100, self, env, python_weights = weights)
            this_task2.run()
            print('Mean length: %f, Mean return: %f' % (np.mean(this_task2.episode_lengths), np.mean(this_task2.episode_returns)))    
            df = pd.DataFrame({'Episode lengths': this_task2.episode_lengths, 'Episode returns': this_task2.episode_returns})
            print(df.describe())

    def double_Q_function_1(self, s): 

        '''
        Used in the case of a double-Q agent
        Takes an input a state and returns a vector of Q-values
        Updates function 1
        '''

        hidden_fixed1 = tf.nn.relu(tf.nn.dropout(tf.matmul(s, self.hidden_weight_old) + self.hidden_bias_old, keep_prob = self.dropout))
        q_fixed1 = tf.matmul(hidden_fixed1, self.linear_weight_old) + self.linear_bias_old
        this_action_fixed1 = tf.argmax(q_fixed1, 1)
        this_action_one_hot1 = tf.one_hot(this_action_fixed1, 2)

        hidden = tf.nn.relu(tf.nn.relu(tf.nn.dropout(tf.matmul(s, self.hidden_weight) + self.hidden_bias, keep_prob = self.dropout)))
        q = tf.matmul(hidden, self.linear_weight) + self.linear_bias
        actual = tf.reduce_sum(tf.multiply(q, this_action_one_hot1),1)

        return actual

    def double_Q_function_2(self, s):

        '''
        Used in the case of a double-Q agent
        Takes an input a state and returns a vector of Q-values
        Updates function 2
        '''

        hidden_fixed2 = tf.nn.relu(tf.nn.dropout(tf.matmul(s, self.hidden_weight_old) + self.hidden_bias_old, keep_prob = self.dropout))
        q_fixed2 = tf.matmul(hidden_fixed2, self.linear_weight_old) + self.linear_bias_old
        this_action_fixed2 = tf.argmax(q_fixed2, 1)
        this_action_one_hot2 = tf.one_hot(this_action_fixed2, 2)

        hidden = tf.nn.relu(tf.nn.relu(tf.nn.dropout(tf.matmul(s, self.hidden_weight_2) + self.hidden_bias_2, keep_prob = self.dropout)))
        q = tf.matmul(hidden, self.linear_weight_2) + self.linear_bias_2
        actual = tf.reduce_sum(tf.multiply(q, this_action_one_hot2),1)

        return actual


    def Q_function(self, s, fixed):

        '''
        Used in the case of a single-Q agent
        Takes an input a state and returns a vector of Q-values
        '''

        if fixed == True:
            if self.nn:
                hidden = tf.nn.relu(tf.nn.dropout(tf.matmul(s, self.hidden_weight_old) + self.hidden_bias_old, keep_prob = self.dropout))
                q = tf.matmul(hidden, self.linear_weight_old) + self.linear_bias_old
            else:
                q = tf.matmul(s, self.linear_weight_old) + self.linear_bias_old
        else:
            if self.nn:
                hidden = tf.nn.relu(tf.nn.relu(tf.nn.dropout(tf.matmul(s, self.hidden_weight) + self.hidden_bias, keep_prob = self.dropout)))
                q = tf.matmul(hidden, self.linear_weight) + self.linear_bias
            else:
                q = tf.matmul(s, self.linear_weight) + self.linear_bias
        return q

    def batch_train(self, data, save_every, test_task = None, task_name = None, name = None):

        '''
        Batch training routine
        Takes as input a train set/ replay buffer
        Trains for self.num_epochs epochs
        Updates the target network every self.num_fixed batches
        Evaluates and saved the model every self.save_every batches
        '''

        for self.epoch in range(self.num_epochs):
            length_data = len(data[0])
            perm = np.random.permutation(length_data)

            data_observations_t = data[0][perm]
            data_actions = data[1][perm]
            data_rewards = data[2][perm]
            data_observations_t_1 = data[3][perm]

            sum_loss = 0
            draw = np.random.random()

            for batch in range(length_data // self.batch_size):

                batch_states_t = data_observations_t[batch * self.batch_size: (batch + 1) * self.batch_size, :]
                batch_states_t_1 =data_observations_t_1[batch * self.batch_size: (batch + 1) * self.batch_size]
                batch_actions = data_actions[batch * self.batch_size: (batch + 1) * self.batch_size]
                batch_rewards = data_rewards[batch * self.batch_size: (batch + 1) * self.batch_size]

                if self.double_Q:

                    if draw < 0.5: # Update function 1
                        feed_dict = {self.states_t: batch_states_t, self.states_t_1: batch_states_t_1, self.actions: batch_actions, 
                        self.rewards: batch_rewards, self.dropout: self.keep_prob_dropout,
                        self.linear_weight_old: self.current_weight_2, self.linear_bias_old: self.current_bias_2, 
                        self.hidden_weight_old: self.current_weight_h_2, self.hidden_bias_old: self.current_bias_h_2}

                        _, this_loss, actions, w, b, w_h, b_h = self.sess.run([self.opt_op1, self.loss1, self.action, 
                            self.linear_weight, self.linear_bias, self.hidden_weight, self.hidden_bias], feed_dict = feed_dict)

                    else: # Update function 2
                        feed_dict = {self.states_t: batch_states_t, self.states_t_1: batch_states_t_1, self.actions: batch_actions, 
                        self.rewards: batch_rewards, self.dropout: self.keep_prob_dropout,
                        self.linear_weight_old: self.current_weight, self.linear_bias_old: self.current_bias, 
                        self.hidden_weight_old: self.current_weight_h, self.hidden_bias_old: self.current_bias_h}

                        _, this_loss, actions, w, b, w_h, b_h = self.sess.run([self.opt_op2, self.loss2, self.action, 
                            self.linear_weight_2, self.linear_bias_2, self.hidden_weight_2, self.hidden_bias_2], feed_dict = feed_dict)

                elif self.nn:
                    feed_dict = {self.states_t: batch_states_t, self.states_t_1: batch_states_t_1, self.actions: batch_actions, self.rewards: batch_rewards, self.dropout: self.keep_prob_dropout, 
                    self.linear_weight_old: self.current_weight, self.linear_bias_old: self.current_bias, 
                    self.hidden_weight_old: self.current_weight_h, self.hidden_bias_old: self.current_bias_h}

                    _, this_loss, actions, w, b, w_h, b_h = self.sess.run([self.opt_op, self.loss, self.action, 
                        self.linear_weight, self.linear_bias, self.hidden_weight, self.hidden_bias], feed_dict = feed_dict)

                else:
                    feed_dict = {self.states_t: batch_states_t, self.states_t_1: batch_states_t_1, self.actions: batch_actions, self.rewards: batch_rewards, self.dropout: self.keep_prob_dropout,
                    self.linear_weight_old: self.current_weight, self.linear_bias_old: self.current_bias}

                    _, this_loss, actions, w, b = self.sess.run([self.opt_op, self.loss, self.action, 
                    self.linear_weight, self.linear_bias], feed_dict = feed_dict)

                sum_loss += this_loss
                
                if (self.global_step.eval(session = self.sess) % self.num_fixed == 0) and ((batch > 0) or (length_data == 1)):

                    self.current_weight = w
                    self.current_bias = b
                    if self.nn:
                        self.current_weight_h = w_h
                        self.current_bias_h = b_h
                    if self.double_Q == True:
                        draw = np.random.random()

                if test_task != None and (self.global_step.eval(session = self.sess) % save_every == 0) and save_every > 0:

                    avg_loss = sum_loss/save_every
                    sum_loss = 0
                    test_task.run()
                    
                    if np.mean(test_task.episode_lengths) >= np.max(self.lengths):
                        self.save_model(task_name, str(self.epoch), self.global_step.eval(session = self.sess))
                        
                    self.lengths = np.concatenate((self.lengths, [np.mean(test_task.episode_lengths)]), 0)
                    self.returns = np.concatenate((self.returns, [np.mean(test_task.episode_returns)]), 0)
                    self.losses = np.concatenate((self.losses, np.array([avg_loss])),0)

    def evaluate(self, env, bool_print = False):
        
        '''
        Takes as input a gym environment
        Creates a task and runs the agent
        Returns the resulting episode lengths and returns
        '''

        this_task = task(10, self, env)
        this_task.run()
        if bool_print:
            df = pd.DataFrame({'Episode lengths': this_task.episode_lengths, 'Episode returns': this_task.episode_returns})
            print('Mean length: %f, Mean return: %f' % (np.mean(this_task.episode_lengths), np.mean(this_task.episode_returns)))
        env.close()

        return this_task.episode_lengths, this_task.episode_returns

    def act(self, observation, reward, done):

        '''
        Lets the agent choose an action during training time
        Takes as input an observations, reward and done flag
        Returns a random action with probability self.epsilon, and the greedy action otherwise
        '''

        observation = np.reshape(observation, [1,4])
        ran = np.random.random()
        if self.nn:
                action = self.sess.run(self.action, feed_dict = {self.states_t: observation, self.dropout: 1.0,
                    self.linear_weight_old: self.current_weight, self.linear_bias_old: self.current_bias, self.hidden_weight_old: self.current_weight_h, self.hidden_bias_old: self.current_bias_h})
        else:
            action = self.sess.run(self.action, feed_dict = {self.states_t: observation, self.dropout: 1.0,
                self.linear_weight_old: self.current_weight, self.linear_bias_old: self.current_bias})
        if ran < self.epsilon:
            return np.mod(action[0] + 1, 2)
        else:
            return action[0]

    # def get_L2_loss(self):
    #     all_vars = tf.trainable_variables() 
    #     return tf.add_n([ tf.nn.l2_loss(v) for v in all_vars if 'bias' not in v.name ]) * self.L2 

    def save_model(self, task, name, epoch):

        if self.nn:
            parameters = self.sess.run([self.hidden_weight, self.hidden_bias, self.linear_weight, self.linear_bias], feed_dict = {})
        else:
            parameters = self.sess.run([self.linear_weight, self.linear_bias], feed_dict = {})
        if not os.path.exists('./models/task_%s_%s/' % (task, name)):
            os.mkdir('./models/task_%s_%s/' % (task, name))
        with open('./models/task_%s_%s/model_epoch_%d.pkl' % (task, name, epoch), 'wb') as f:
            pkl.dump(parameters, f)

class task(object):

    def __init__(self, num_episodes, agent, env, max_episode_steps = 300, discount = 0.99, online = False, replay = False, iteration = None, eval_task = None, python_weights = None):
        self.num_episodes = num_episodes
        self.agent = agent
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.discount = discount
        self.online = online
        self.iteration = iteration
        self.eval_task = eval_task
        self.python_weights = python_weights
        

    def reward_wrapper(self, d):

        '''
        Clips rewards to -1 (if the pole falls) and 0 (otherwise)
        '''

        if d:
            return -1
        else:
            return 0

    def relu(self, array):
        return np.maximum(0,array)

    def python_act(self, observation):

        '''
        Lets the agent choose an action during test time
        Takes as input an observation
        Returns the greedily selected action
        '''

        if len(self.python_weights) == 4:
            hw = self.python_weights[0]
            hb = self.python_weights[1]
            lw = self.python_weights[2]
            lb = self.python_weights[3]
            observation = np.reshape(observation, [1,4])
            hidden = self.relu(np.dot(observation, hw) + hb)
            q = np.dot(hidden, lw) + lb
        else:
            lw = self.python_weights[0]
            lb = self.python_weights[1]
            q = np.dot(observation, lw) + lb
        action = np.argmax(q)
        return action

    def run(self):

        self.episode_returns = []
        self.episode_lengths = []
        self.episode_actions = []
        self.episode_observations = []
        self.episode_rewards = []

        for i_episode in range(self.num_episodes):

            observation = self.env.reset()

            reward = 0
            done = False
            episode_return = 0
            step_observations = [observation]
            step_actions = []
            step_rewards = []

            for t in range(self.max_episode_steps):
                if self.python_weights == None:
                    action = self.agent.act(observation, reward, done)
                else:
                    action = self.python_act(observation)

                observation, _, done, _ = self.env.step(action)
                reward = self.reward_wrapper(done)
                episode_return += reward * self.discount**t 

                step_actions.append(action)
                step_rewards.append(reward)
                step_observations.append(observation)

                if self.online:
                    data = [np.expand_dims(np.asarray(step_observations[t]),0), np.asarray([action]), np.asarray([reward]), np.expand_dims(np.asarray(step_observations[t+1]),0)]
                    self.agent.batch_train(data, 0)
                if done:
                    break
                
            self.episode_actions.append(step_actions)
            self.episode_rewards.append(step_rewards)    
            self.episode_observations.append(step_observations)
            self.episode_returns.append(episode_return)
            self.episode_lengths.append(t+1)

    def package_data(self):
        data_actions = []
        data_rewards = []
        data_observations = []
        data_observations_t = []

        for i in range(len(self.episode_actions)):
            for j in range(len(self.episode_actions[i])):
                data_actions.append(self.episode_actions[i][j])
                data_rewards.append(self.episode_rewards[i][j])
                data_observations.append(self.episode_observations[i][j])
                data_observations_t.append(self.episode_observations[i][j+1])
        data = [np.asarray(data_observations), np.asarray(data_actions), np.asarray(data_rewards), np.asarray(data_observations_t)]

        return data

if __name__ == '__main__':

    def task_a_1():
        env = gym.make('CartPole-v0')
        env._max_episode_steps = 300
        agent_R = RandomAgent(env.action_space)
        task_a_1 = task(3, agent_R, env)
        task_a_1.run()
        df_a_1 = pd.DataFrame({'Episode lengths': task_a_1.episode_lengths, 'Episode returns': task_a_1.episode_returns})
        print(df_a_1.describe())
        print(df_a_1)
        env.close()

    def task_a_2():
        env = gym.make('CartPole-v0')
        env._max_episode_steps = 300
        agent_R = RandomAgent(env.action_space)
        task_a_2 = task(100, agent_R, env)
        task_a_2.run()
        df_a_2 = pd.DataFrame({'Episode lengths': task_a_2.episode_lengths, 'Episode returns': task_a_2.episode_returns})
        print(df_a_2.describe())
        env.close()

    def task_a_3(nn, mode, load):

        env = gym.make('CartPole-v0')
        env._max_episode_steps = 300
        agent_R = RandomAgent(env.action_space)
        task_a_3 = task(2000, agent_R, env)
        task_a_3.run()

        data = task_a_3.package_data()
        
        
        if mode == 'single':
            with tf.Session() as sess:
                if nn:
                    agent_Q = AgentQ(env.action_space, 0.001, optimizer = 'RMSProp', nn = True, 
                        batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, num_fixed = 1, clip = None, decay = 0.99, load_path = load, external_session = sess)
                else:
                    agent_Q = AgentQ(env.action_space, 0.001, optimizer = 'RMSProp', units = 4, nn = False, 
                        batch_size = 64, num_fixed = 10, clip = None, L2 = 0.0, decay = 1.0, load_path = load, external_session = sess)
                
                task_test = task(10, agent_Q, env)
                if load == None:  
                    agent_Q.batch_train(data, 50, task_test, 'a3', str(agent_Q.learning_rate))
            

        else:
            learning_rates = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 0.5]

            if not os.path.isdir('./plots/a3/'):
                os.mkdir('./plots/a3/')

            for rate in learning_rates:
                tf.reset_default_graph()

                with tf.Session() as sess:
                    print('Running now with rate', rate)
                    if nn:
                        agent_Q = AgentQ(env.action_space, eta = rate, optimizer = 'RMSProp', units = 100, nn = True, relu= True, num_epochs = 5, 
                            batch_size = 32, keep_prob_dropout = 0.9, L2 = 0, num_fixed = 1, clip = None, decay = 0.95, decay_steps = 100, external_session = sess)
                        name = str('a3_nn_%s' % str(rate))
                    else:
                        agent_Q = AgentQ(env.action_space, eta = rate, optimizer = 'RMSProp', units = 4, nn = False, num_epochs = 5, 
                            batch_size = 32, num_fixed = 1, clip = None, L2 = 0.0, decay = 0.95, decay_steps = 100, external_session = sess)
                        name = str('a3_linear_%s' % str(rate))

                    task_test = task(10, agent_Q, env)
                    agent_Q.batch_train(data, 50, task_test, name, str(rate))

                    print('now plot')
                    fig = plt.subplots(nrows=3, ncols=1)
                    plt.tight_layout()
                    p = plt.subplot(3, 1, 1)
                    p.set_title("Average discounted episode return", fontsize = 8)
                    p.plot(agent_Q.returns)
                    p.set_ylim(-1, 0)
                    p = plt.subplot(3, 1, 2)
                    p.set_title("Average episode length", fontsize = 8)
                    p.plot(agent_Q.lengths)
                    p.set_ylim(0, 300)
                    p = plt.subplot(3, 1, 3)
                    p.set_title("Average loss over 50 batches", fontsize = 8)
                    p.plot(agent_Q.losses)
                    plt.savefig('./plots/a3/%s_%s.png' % (name, str(rate)))
                    plt.close()
                    sess.close()
                

    def task_a_4(load_path = None):
        env = gym.make('CartPole-v0')
        env._max_episode_steps = 300
        
        episodes = 2000
        runs = 100
        output_lengths = np.zeros([runs, episodes, 1])
        output_returns = np.zeros([runs, episodes, 1])

        if load_path != None:
            tf.reset_default_graph()
            with tf.Session() as sess:
                agent_Q_online = AgentQ(env.action_space, 0.001, optimizer = 'RMSProp', units = 100, nn = True, relu= True, num_epochs = 1, 
                            batch_size = 1, keep_prob_dropout = 0.9, L2 = 0, num_fixed = 1, clip = None, epsilon = 0.05, external_session = sess, load_path = load_path)

        else:
            for i in range(runs):
                tf.reset_default_graph()
                with tf.Session() as sess:

                    agent_Q_online = AgentQ(env.action_space, 0.0005, optimizer = 'RMSProp', units = 100, nn = True, relu= True, num_epochs = 1, 
                        batch_size = 1, keep_prob_dropout = 0.9, L2 = 0, num_fixed = 1, clip = None, epsilon = 0.05, external_session = sess)
                    
                    max_length = 0

                    for trial in range(episodes):
                        task_a_4 = task(1, agent_Q_online, env, online=True, iteration = i)
                        task_a_4.run()
                        this_length, this_return = agent_Q_online.evaluate(env)
                        output_lengths[i, trial] = np.mean(this_length)
                        output_returns[i, trial] = np.mean(this_return)
                        print('\rRun: %d, Episode: %d, Mean length: %f, Mean return: %f' % (i, trial, np.mean(this_length), np.mean(this_return)), end='')
                        
                        if np.mean(this_length) > max_length:
                            agent_Q_online.save_model('a4', str(i), trial)
                            max_length = np.mean(this_length)


                    mean_lengths = np.sum(output_lengths, 0)/(i+1)
                    mean_returns = np.sum(output_returns, 0)/(i+1)

                    fig = plt.subplots(nrows=2, ncols=1)
                    plt.tight_layout()
                    p = plt.subplot(2, 1, 1)
                    p.set_title("Average discounted episode return", fontsize = 8)
                    p.plot(mean_returns)
                    p.set_ylim(-1, 0)
                    p = plt.subplot(2, 1, 2)
                    p.set_title("Average episode length", fontsize = 8)
                    p.plot(mean_lengths)
                    p.set_ylim(0, 300)
                    plt.savefig('./plots/a4/plot.png')
                    plt.close()

    def task_a_5(load_path = None):
        env = gym.make('CartPole-v0')
        env._max_episode_steps = 300
        
        episodes = 2000
        runs = 1
        output_lengths = np.zeros([runs, episodes, 1])
        output_returns = np.zeros([runs, episodes, 1])

        if load_path != None:
            with tf.Session() as sess:
                agent_Q_online = AgentQ(env.action_space, 0.001, optimizer = 'RMSProp', units = 1000, nn = True, relu= True, num_epochs = 1, 
                            batch_size = 1, keep_prob_dropout = 0.9, L2 = 0, num_fixed = 1, clip = None, epsilon = 0.05, external_session = sess, load_path = load_path)

        else:
            for i in range(runs):
                tf.reset_default_graph()
                with tf.Session() as sess:

                    agent_Q_online = AgentQ(env.action_space, 0.0005, optimizer = 'RMSProp', units = 30, nn = True, relu= True, num_epochs = 1, 
                        batch_size = 1, keep_prob_dropout = 0.9, L2 = 0, num_fixed = 1, clip = None, epsilon = 0.05, external_session = sess)
                    
                    max_length = 0

                    for trial in range(episodes):
                        task_a_4 = task(1, agent_Q_online, env, online=True, iteration = i)
                        task_a_4.run()
                        this_length, this_return = agent_Q_online.evaluate(env)
                        output_lengths[i, trial] = np.mean(this_length)
                        output_returns[i, trial] = np.mean(this_return)
                        print('\rRun: %d, Episode: %d, Mean length: %f, Mean return: %f' % (i, trial, np.mean(this_length), np.mean(this_return)), end='')
                        
                        if np.mean(this_length) > max_length:
                            agent_Q_online.save_model('a5', str(i), trial)
                            max_length = np.mean(this_length)


                    mean_lengths = np.sum(output_lengths, 0)/(i+1)
                    mean_returns = np.sum(output_returns, 0)/(i+1)
                    df = pd.DataFrame({'Episode lengths': mean_lengths[:,0], 'Episode returns': mean_returns[:,0]})
                    print(df.describe())

                    fig = plt.subplots(nrows=2, ncols=1)
                    plt.tight_layout()
                    p = plt.subplot(2, 1, 1)
                    p.set_title("Average discounted episode return", fontsize = 8)
                    p.plot(mean_returns)
                    p.set_ylim(-1, 0)
                    p = plt.subplot(2, 1, 2)
                    p.set_title("Average episode length", fontsize = 8)
                    p.plot(mean_lengths)
                    p.set_ylim(0, 300)
                    plt.savefig('./plots/a5/plot.png')
                    plt.close()


    def task_a_6(load_path = None):
        env = gym.make('CartPole-v0')
        env._max_episode_steps = 300

        if load_path != None:
            with tf.Session() as sess:
                agent_Q_replay = AgentQ(env.action_space, 0.001, optimizer = 'RMSProp', units = 100, nn = True, relu= True, num_epochs = 1, 
                    batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, num_fixed = 1, clip = None, decay = 0.99, epsilon = 0.05, external_session = sess, load_path = load_path)

        else:
            with tf.Session() as sess:
                agent_Q_replay = AgentQ(env.action_space, 0.001, optimizer = 'RMSProp', units = 100, nn = True, relu= True, num_epochs = 1, 
                    batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, num_fixed = 1, clip = None, decay = 0.99, epsilon = 0.05, external_session = sess)
                task_test = task(10, agent_Q_replay, env)

                for trial in range(10):
                    # print('\nEpisode:', trial, end='')
                    task_a_6 = task(300, agent_Q_replay, env)
                    task_a_6.run()

                    if trial == 0:
                        replay_buffer = task_a_6.package_data()

                    else:
                        trial_data = task_a_6.package_data()
                        data_observations = np.concatenate((replay_buffer[0], trial_data[0]), 0)[-50000:]
                        data_actions = np.concatenate((replay_buffer[1], trial_data[1]), 0)[-50000:]
                        data_rewards = np.concatenate((replay_buffer[2], trial_data[2]), 0)[-50000:]
                        data_observations_t = np.concatenate((replay_buffer[3], trial_data[3]), 0)[-50000:]
                        replay_buffer = [data_observations, data_actions, data_rewards, data_observations_t]

                    agent_Q_replay.batch_train(replay_buffer, 20, task_test, 'a6')
                    this_length, this_return = agent_Q_replay.evaluate(env)
                    print('\rEpisode: %d, Mean length: %f, Mean return: %f' % (trial, np.mean(this_length), np.mean(this_return)), end='')
                
                    fig = plt.subplots(nrows=3, ncols=1)
                    plt.tight_layout()
                    p = plt.subplot(3, 1, 1)
                    p.set_title("Average discounted episode return", fontsize = 8)
                    p.plot(agent_Q_replay.returns)
                    p.set_ylim(-1, 0)
                    p = plt.subplot(3, 1, 2)
                    p.set_title("Average episode length", fontsize = 8)
                    p.plot(agent_Q_replay.lengths)
                    p.set_ylim(0, 300)
                    p = plt.subplot(3, 1, 3)
                    p.set_title("Average loss over 50 batches", fontsize = 8)
                    p.plot(agent_Q_replay.losses)
                    plt.savefig('./plots/a6/plot.png')
                    plt.close()
                sess.close()

        
    def task_a_7(buffer_size, load_path = None):
        env = gym.make('CartPole-v0')
        env._max_episode_steps = 300

        if load_path != None:
            with tf.Session() as sess:
                agent_Q_replay = AgentQ(env.action_space, 0.001, optimizer = 'RMSProp', units = 100, nn = True, relu= True, num_epochs = 1, 
                    batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, num_fixed = 5, clip = None, decay = 1.0, epsilon = 0.05, external_session = sess, load_path = load_path)

        else:
            with tf.Session() as sess:
                agent_Q_replay = AgentQ(env.action_space, 0.001, optimizer = 'RMSProp', units = 100, nn = True, relu= True, num_epochs = 1, 
                    batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, num_fixed = 5, clip = None, decay = 0.99, epsilon = 0.05, external_session = sess, load_path = load_path)
                task_test = task(10, agent_Q_replay, env)

                for trial in range(20):
                    task_a_7 = task(300, agent_Q_replay, env)
                    task_a_7.run()

                    if trial == 0:
                        replay_buffer = task_a_7.package_data()

                    else:
                        trial_data = task_a_7.package_data()
                        data_observations = np.concatenate((replay_buffer[0], trial_data[0]), 0)[-buffer_size:]
                        data_actions = np.concatenate((replay_buffer[1], trial_data[1]), 0)[-buffer_size:]
                        data_rewards = np.concatenate((replay_buffer[2], trial_data[2]), 0)[-buffer_size:]
                        data_observations_t = np.concatenate((replay_buffer[3], trial_data[3]), 0)[-buffer_size:]
                        replay_buffer = [data_observations, data_actions, data_rewards, data_observations_t]

                    agent_Q_replay.batch_train(replay_buffer, 20, task_test, 'a7')
                    this_length, this_return = agent_Q_replay.evaluate(env)
                    print('\rEpisode: %d, Mean length: %f, Mean return: %f, buffer size: %d' % (trial, np.mean(this_length), np.mean(this_return), replay_buffer[0].shape[0]), end='')
                
                    fig = plt.subplots(nrows=3, ncols=1)
                    plt.tight_layout()
                    p = plt.subplot(3, 1, 1)
                    p.set_title("Average discounted episode return", fontsize = 8)
                    p.plot(agent_Q_replay.returns)
                    p.set_ylim(-1, 0)
                    p = plt.subplot(3, 1, 2)
                    p.set_title("Average episode length", fontsize = 8)
                    p.plot(agent_Q_replay.lengths)
                    p.set_ylim(0, 300)
                    p = plt.subplot(3, 1, 3)
                    p.set_title("Average loss over 50 batches", fontsize = 8)
                    p.plot(agent_Q_replay.losses)
                    plt.savefig('./plots/a7/plot.png')
                    plt.close()
                sess.close()

    def task_a_8(buffer_size, load_path = None):
        env = gym.make('CartPole-v0')
        env._max_episode_steps = 300

        if load_path != None:
            with tf.Session() as sess:
                agent_Q_replay = AgentQ(env.action_space, 0.001, optimizer = 'RMSProp', units = 100, nn = True, relu= True, num_epochs = 1, 
                    batch_size = 32, keep_prob_dropout = 1.0, L2 = 0, num_fixed = 1, clip = None, decay = 0.99, epsilon = 0.05, double_Q = True, external_session = sess, load_path = load_path)
        
        else:
            with tf.Session() as sess:
                agent_Q_replay = AgentQ(env.action_space, 0.001, optimizer = 'RMSProp', units = 100, nn = True, relu= True, num_epochs = 1, 
                    batch_size = 32, keep_prob_dropout = 0.9, L2 = 0, num_fixed = 5, clip = None, decay_steps = 10000, decay = 0.99, epsilon = 0.05, double_Q = True, external_session = sess, load_path = load_path)
                task_test = task(10, agent_Q_replay, env)

                for trial in range(20):
                    task_a_8 = task(300, agent_Q_replay, env)
                    task_a_8.run()

                    if trial == 0:
                        replay_buffer = task_a_8.package_data()

                    else:
                        trial_data = task_a_8.package_data()
                        perm = np.random.permutation(len(replay_buffer[0])+len(trial_data[0]))

                        data_observations = np.concatenate((replay_buffer[0], trial_data[0]), 0)[perm][-buffer_size:]
                        data_actions = np.concatenate((replay_buffer[1], trial_data[1]), 0)[perm][-buffer_size:]
                        data_rewards = np.concatenate((replay_buffer[2], trial_data[2]), 0)[perm][-buffer_size:]
                        data_observations_t = np.concatenate((replay_buffer[3], trial_data[3]), 0)[perm][-buffer_size:]
                        # data_observations = np.concatenate((replay_buffer[0], trial_data[0]), 0)[-buffer_size:]
                        # data_actions = np.concatenate((replay_buffer[1], trial_data[1]), 0)[-buffer_size:]
                        # data_rewards = np.concatenate((replay_buffer[2], trial_data[2]), 0)[-buffer_size:]
                        # data_observations_t = np.concatenate((replay_buffer[3], trial_data[3]), 0)[-buffer_size:]
                        # replay_buffer = [data_observations, data_actions, data_rewards, data_observations_t]
                        # data_observations = np.concatenate((replay_buffer[0], trial_data[0]), 0)[5000:]
                        # data_actions = np.concatenate((replay_buffer[1], trial_data[1]), 0)[5000:]
                        # data_rewards = np.concatenate((replay_buffer[2], trial_data[2]), 0)[5000:]
                        # data_observations_t = np.concatenate((replay_buffer[3], trial_data[3]), 0)[5000:]
                        replay_buffer = [data_observations, data_actions, data_rewards, data_observations_t]

                    agent_Q_replay.batch_train(replay_buffer, 20, task_test, 'a8')
                    this_length, this_return = agent_Q_replay.evaluate(env)
                    print('\rEpisode: %d, Mean length: %f, Mean return: %f, buffer size: %d' % (trial, np.mean(this_length), np.mean(this_return), replay_buffer[0].shape[0]), end='')
                
                    fig = plt.subplots(nrows=3, ncols=1)
                    plt.tight_layout()
                    p = plt.subplot(3, 1, 1)
                    p.set_title("Average discounted episode return", fontsize = 8)
                    p.plot(agent_Q_replay.returns)
                    p.set_ylim(-1, 0)
                    p = plt.subplot(3, 1, 2)
                    p.set_title("Average episode length", fontsize = 8)
                    p.plot(agent_Q_replay.lengths)
                    p.set_ylim(0, 300)
                    p = plt.subplot(3, 1, 3)
                    p.set_title("Average loss over 50 batches", fontsize = 8)
                    p.plot(agent_Q_replay.losses)
                    plt.savefig('./plots/a8/plot.png')
                    plt.close()
                sess.close()

    # task_a_1()
    # task_a_2()

    # Uncomment to load and test models
    # task_a_3(nn = False, mode ='single', load = '../models/task_a3/task_a3_linear_1e-05/model.pkl')
    # task_a_3(nn = False, mode ='single', load = '../models/task_a3/task_a3_linear_0.0001/model.pkl')
    # task_a_3(nn = False, mode ='single', load = '../models/task_a3/task_a3_linear_0.001/model.pkl')
    # task_a_3(nn = False, mode ='single', load = '../models/task_a3/task_a3_linear_0.01/model.pkl')
    # task_a_3(nn = False, mode ='single', load = '../models/task_a3/task_a3_linear_0.1/model.pkl')
    # task_a_3(nn = False, mode ='single', load = '../models/task_a3/task_a3_linear_0.5/model.pkl')
    # task_a_3(nn = True, mode ='single', load = '../models/task_a3/task_a3_nn_1e-05/model.pkl')
    # task_a_3(nn = True, mode ='single', load = '../models/task_a3/task_a3_nn_0.0001/model.pkl')
    # task_a_3(nn = True, mode ='single', load = '../models/task_a3/task_a3_nn_0.001/model.pkl')
    # task_a_3(nn = True, mode ='single', load = '../models/task_a3/task_a3_nn_0.01/model.pkl')
    # task_a_3(nn = True, mode ='single', load = '../models/task_a3/task_a3_nn_0.1/model.pkl')
    # task_a_3(nn = True, mode ='single', load = '../models/task_a3/task_a3_nn_0.5/model.pkl')
    # task_a_4(load_path = '../models/task_a4/task_a4_2/model.pkl')
    # task_a_5(load_path = '../models/task_a5/model_30.pkl')§
    # task_a_5(load_path = '../models/task_a5/model_1000.pkl')
    # task_a_6(load_path = '../models/task_a6/model.pkl')
    task_a_7(50000, load_path = '../models/task_a7/model.pkl')
    # task_a_8(50000, load_path = '../models/task_a8/model.pkl')
    
    # Uncomment to train models
    # task_a_3(nn = True)
    # task_a_4()
    # task_a_5()
    # task_a_6()
    # task_a_7(50000)
    # task_a_8(50000)