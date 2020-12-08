# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from time import time

from LineEnv import LineEnv

class NetAgent:
    def __init__(self, name, save_folder = ''):
        self.env = LineEnv(name)
        self.history = []
        self.name = name
        self.save_folder = save_folder
        self.settings = {}

    def discount_rewards(self, rewards, discount_rate):
        discounted_rewards = np.empty(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards*discount_rate
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards
    
    def discount_and_normalize_rewards(self, all_rewards, discount_rate):
        all_discounted_rewards = [self.discount_rewards(rewards, discount_rate) for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        
        return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]
    
    def train_net(self, 
                  n_iterations = 200,       # Training iterations
                  n_max_steps = 1400,       # Steps per episode
                  n_games_per_episode = 20, # Train policy every game
                  discount_rate = 0.98,     # Credit assignment
                  pretrained = "",
                  save = True):
        # Architecture
        n_inputs = 4 # (pos_x, pos_y, vel_x, vel_y)
        n_hidden = 10
        n_outputs = 1
        learning_rate = 0.012
        initializer = tf.contrib.layers.variance_scaling_initializer()
        
        # Building. We stick to a single hidden layer.
        X = tf.placeholder(tf.float32, shape=[None, n_inputs])
        hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.relu, kernel_initializer=initializer)
        logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
        outputs = tf.nn.sigmoid(logits)
        
        # Sample multinomial distrib to pick action given probabilities
        p_thrust_nothrust = tf.concat(axis=1, values=[outputs, 1-outputs])
        action = tf.multinomial(tf.log(p_thrust_nothrust), num_samples=1)
        
        # Target - we pretend the chosen action is the best possible action.
        # Down the line, we evaluate the (discounted) scores obtained since this action was chosen.
        # If that score beats other actions, then we train the net to "like" this action by applying the corresponding gradient.
        y = 1. - tf.to_float(action)
        
        # Cost function and optimizer
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = logits)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        
        # Store gradients for possible future application
        grads_and_vars = optimizer.compute_gradients(cross_entropy)
        gradients = [grad for grad, var in grads_and_vars]
        gradient_placeholders = []
        grads_and_vars_feed = []
        for grad, variable in grads_and_vars:
            gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
            gradient_placeholders.append(gradient_placeholder)
            grads_and_vars_feed.append((gradient_placeholder, variable))
            
        training_op = optimizer.apply_gradients(grads_and_vars_feed)
        
        init = tf.global_variables_initializer()
        saver = tf.compat.v1.train.Saver()
        # End of graph construction

        # Let's train        
        with tf.Session() as sess:
            if not pretrained:
                init.run()
            else:
                saver.restore(sess, pretrained)
            for iteration in tqdm(range(n_iterations)):
                all_rewards = []
                all_gradients = []
                
                for game in range(n_games_per_episode):
                    current_rewards = []
                    current_gradients = []
                    obs = self.env.Reset()
                    
                    for step in range(n_max_steps):
                        # s = time()
                        action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
                        # print(f'tf: {time()-s}')
                        # s = time()
                        obs, reward, done = self.env.Step(action_val[0][0])
                        # print(f'step: {time()-s}')
                        current_rewards.append(reward)
                        current_gradients.append(gradients_val)
                        if done:
                            break
                        
                    all_rewards.append(current_rewards)
                    all_gradients.append(current_gradients)
                
                # Episode complete. Train net
                all_rewards = self.discount_and_normalize_rewards(all_rewards, discount_rate)
                feed_dict = {}
                for var_index, grad_placeholder in enumerate(gradient_placeholders):
                    mean_gradients = np.mean(
                        [reward*all_gradients[game_index][step][var_index]
                         for game_index, rewards in enumerate(all_rewards)
                         for step, reward in enumerate(rewards)],
                        axis = 0
                        )
                    feed_dict[grad_placeholder] = mean_gradients
                sess.run(training_op, feed_dict = feed_dict)
                
                self.history.append(self.env.history)
                
            # Done training, save model
            if save:
                saver.save(sess, self.save_folder + self.name +'.ckpt')
        self.settings = {
            'n_iterations': n_iterations,
            'n_max_steps': n_max_steps,
            'n_games_per_episode': n_games_per_episode,
            'discount_rate': discount_rate,
            'n_inputs': n_inputs,
            'n_hidden': n_hidden,
            'learning_rate': learning_rate
        }

save_folder = 'NetAgentModels/'
agent = 'Smith_v8'
na = NetAgent(agent, save_folder = save_folder)
na.train_net(pretrained = save_folder + 'Smith_v7.ckpt')

pos = list(zip(*na.env.history['positions']))
score = int(sum(map(na.env.CalculateReward, na.env.history['positions'])))
# How did we do?
# print(f'Total reward: {sum(all_rewards)}') 
# What about the raw score?
# print(f'Raw reward: {')


# Draw trajectory

plt.plot(pos[1])
plt.title(f'Agent {agent}. Score {score}')
# plt.title(f'Score: {env.Score()}\nPolicy: {policy.__name__}')

# actions = np.array(actions) * max(pos[1])
# plt.plot(actions, alpha=.3, color = 'black')
plt.show()


np.save(save_folder + agent + '.settings.npy', na.settings)
np.save(save_folder + agent + '.history.npy', np.array(na.history))