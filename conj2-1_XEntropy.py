'''
In AZ Wagner's paper 'Constructions in combinatorics via neural networks,' he uses 
the cross-entropy RL method to find a 19-vertex counterexample to the following:

Conjecture 2.1: Let G be a connected graph on n >= 3 vertices, with largest 
eigenvalue \lambda_1 and matching number \mu. Then \lambda_1 + \mu >= \sqrt{n-1} + 1.

This is my own code written to recreate his result. With a higher learning
rate (>.0005), the program usually doesn't converge to the desired counterexample. 
This approach is prone to overfitting and converging to a non-counterexample. 
Ideally, with dropout and the other given hyperparameters, the model should 
learn "better", although more slowly.

Further, in the paper "Reinforcment learning for graph theory" by Ghebleh, et al.,
the authors expanded on Wagner's approach by increasing the probability of random 
actions when the algorithm appears to be stuck in a local minima.
This strategy is quite powerful, and thus I have utilized it here with a slightly
different approach.
'''
from collections import namedtuple  # Helps us write cleaner code
from tensorboardX import SummaryWriter     # To help visualize our runs
import matplotlib.pyplot as plt     # For drawing our graphs
import networkx as nx   # Graph package
import numpy as np      # Helpful for some computations
import math             # Helpful for other computations
import os               
import datetime
# PyTorch for deep learning:
import torch
import torch.nn as nn
import torch.optim as optim

N = 19                  # Number of vertices
HIDDEN_SIZE_1 = 128     # Number of neurons in first layer
HIDDEN_SIZE_2 = 64      # Number of neurons in second layer
HIDDEN_SIZE_3 = 4      # Number of neurons in second layer
BATCH_SIZE = 200        # Number of graphs 'played' in each batch
PERCENTILE = 92         # Only the top (100 - PERCENTILE)% graphs in each batch are used to train the NN
SUPER_EPISODES = 3      # The number of super episodes that are saved indefinitely to train on
LEARNING_RATE = 0.0002  # Based on experiments, it seems to learn better on lower learning rates
PLOT_RATE = 500         # Plot the best graph every PLOT_RATE iterations
INF = 2**16             # Placeholder
DROPOUT = 0.2           # Dropout, to avoid overfitting

# More hyperparameters to incorporate random actions when we get stuck in local minima
RAND_ACT_PROB_INIT = 0.001      # Starting probability of random actions
RAND_ACT_PROB_MAX = 0.025       # Maximum random action probability
RAND_ACT_PROB_MULT = 2          # How much we multiply random action probability by when needed
RAND_ACT_PROB_DVSR = 1.05       # How much we divide random action probability by when reward changes
RAND_ACT_PROB_WAIT = 10         # Number of iterations at same reward before increasing random actions

class GraphEnv():
    ''' A graph environment that utilizes the one-hot encoding method used by AZ Wagner
            -- intended to partially emulate a gymnasium environment '''
    def __init__(self,  reward_fn, num_verts = N):
        self.n = num_verts
        self.m = int((self.n * (self.n - 1))/2)
        # We will keep the 'flattened' version of state. See 4-vertex example:
        #        edge (2,3) to be decided next 
        #               v
        #    [[0, 1, 1, 0, 0, 0],  <- records edges; (1,2), (1,3), (1,4) already decided
        #     [0, 0, 0, 1, 0, 0]]  <- one-hot bit
        self.state = torch.zeros((1, 2*self.m), dtype = torch.float) 
        self.cur_edge = 0                   # Index of the edge we are deciding
        self.state[0, self.cur_edge + self.m] = 1     # Our one-hot bit
        self.terminated = False
        self.rew = 0.0
        self.reward_function = reward_fn  # We can pass any reward function to this environment

    def reset(self):
        self.state.zero_()
        self.cur_edge = 0
        self.state[0, self.m] = 1 
        self.terminated = False
        self.rew = 0.0

        return self.state

    def step(self, action):
        if not self.terminated:
            self.state[0, self.cur_edge] = action
            self.state[0, self.cur_edge + self.m] = 0  # Turn off one-hot bit
            # For X-Entropy, the following need only be called after episode terminated
            # self.reward += self.reward_function(self.state)
            self.cur_edge += 1

            if self.cur_edge >= self.m: # Whole graph determined
                self.terminated = True
                # We only update the reward once the whole graph is determined b/c X-entropy only
                # needs the reward to select the elite episodes from the batch
                self.rew = self.reward_function(self.state, self.n)
            else:
                self.state[0, self.cur_edge + self.m] = 1    # Push one-hot bit along
        
        # Gym environments return two other extra info lists, we don't...
        # Other note: we reshape state below to what the NN wants to see
        return self.state, self.rew, self.terminated


def conj21_rew_fn(state, num_verts = N):
    ''' The reward function for Conjecture 2.1, see Wagner paper  '''
    G = nx.Graph()
    G.add_nodes_from(list(range(num_verts)))
    count = 0

    for i in range(num_verts):      # Unpacking the edges from state
        for j in range(i+1, num_verts):
            if state[0, count] == 1:
                G.add_edge(i,j)
            count += 1
    
    if not nx.is_connected(G):
        return -INF
    
    # Calculations for evaluating conjecture
    evals = np.linalg.eigvalsh(nx.adjacency_matrix(G).todense())
    evalsRealAbs = np.zeros_like(evals)
    for i in range(len(evals)):
        evalsRealAbs[i] = abs(evals[i])
    lambda1 = max(evalsRealAbs)
    maxMatch = nx.max_weight_matching(G)
    mu = len(maxMatch)
    # Conjecture 2.1 broken iff score > 0:
    score = math.sqrt(num_verts-1) + 1 - lambda1 - mu

    if score > 0:  
        print('Counterexample Found!', state[0])

    return score


def plot_graph(state, file_name, num_verts = N):
    ''' Plots graphs with MatPlotLib '''
    if state == None:   # In case there is nothing in state
        return 

    G = nx.Graph()
    G.add_nodes_from(list(range(num_verts)))
    count = 0

    for i in range(num_verts):
        for j in range(i+1, num_verts):
            if state[count] == 1:
                G.add_edge(i,j)
            count += 1

    plt.figure()
    nx.draw_kamada_kawai(G, node_size=40)
    plt.savefig(file_name, dpi = 300)

        
class Net(nn.Module):
    ''' Standard creation of a NN with PyTorch '''
    def __init__(self, in_size, out_size, layer_1 = HIDDEN_SIZE_1, layer_2 = HIDDEN_SIZE_2, layer_3 = HIDDEN_SIZE_3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, layer_1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(layer_1, layer_2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(layer_2, layer_3),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(layer_3, out_size)
        )

    def forward(self, x):
        return self.net(x)
    
# The following are used to write more readable code --  these are new tuple subclasses
Episode = namedtuple('Episode', field_names=['reward', 'result', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, random_action_probability, batch_size = BATCH_SIZE):
    ''' Generates batches of graphs '''
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sig = nn.Sigmoid()      # Turns outputs into probabilities, (range is (0,1))
    rap = random_action_probability[0] # Used a list so that we can edit its content between yields

    while  True:
        act_probs = sig(net(obs))

        if np.random.rand() < rap:  # Perform random action if below random action probability
            action = np.random.randint(2)   # Either 0 or 1
        elif np.random.rand() < act_probs:  
            action = 1      # The larger act_probs, the more likely to add the edge
        else:
            action = 0

        step = EpisodeStep(observation = obs.clone(), action = action) # cloning to avoid aliasing issues
        next_obs, reward, terminated = env.step(action)
        episode_reward += reward
        episode_steps.append(step)

        if terminated:
            e = Episode(reward = episode_reward, result = next_obs.clone(), steps = episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()

            if len(batch) == batch_size:
                yield batch
                rap = random_action_probability[0] # Updating our random action probability 
                batch = []

        obs = next_obs


def elites_from_batch(batch, percentile, forbidden_reward):
    ''' Filters a batch, returning the best episodes '''
    rewards = list(map(lambda s: s.reward, batch)) # Extracts all rewards from the batch
    reward_bound = np.percentile(rewards, percentile)  # The min reward required to use for training
    reward_max = float(np.max(rewards))

    train_obs = []
    train_act = []

    batch_empty = True    # Possible no connected graphs in batch, so there is nothing to train on
    best_episode = None   # We will want to return the best graph so that we can plot it

    for episode in batch:
        reward, result, steps = episode
        if reward == reward_max:    # We will save the best episode for later uses
            best_episode = episode
        
        if reward < reward_bound or reward == -INF or abs(reward - forbidden_reward) < 0.001:
            continue

        train_obs.extend(map(lambda step: step.observation.tolist()[0], steps))
        train_act.extend(map(lambda step: [step.action], steps))
        batch_empty = False   # We have at least one 'elite' graph in batch
       
    return train_obs, train_act, reward_max, best_episode, batch_empty


def update_super_episodes(super_episodes, episode_to_add, num_supers = SUPER_EPISODES):
    ''' Maintains the list of super episodes that we keep for training '''
    super_episodes.append(episode_to_add)
    super_episodes.sort(key = lambda episode: episode.reward)

    if len(super_episodes) > num_supers:
        super_episodes = super_episodes[-num_supers:]

    return super_episodes[0].reward


def get_training_batch(batch, percentile, forbidden_reward, super_episodes, num_supers, super_min_reward):
    ''' Primarily, this combines elites from batch with super episodes into tensors for training '''
    train_obs, train_act, reward_max, best_episode, batch_empty = elites_from_batch(batch, percentile, forbidden_reward)
    if reward_max > super_min_reward:
       super_min_reward = update_super_episodes(super_episodes, best_episode, num_supers) 

    no_graphs = batch_empty     # Tracks if there will be anything to train on

    for episode in super_episodes:      # Add data from super episodes
        if episode.reward > forbidden_reward:       # Only want to train on 'better' graphs
            train_obs.extend(map(lambda step: step.observation.tolist()[0], episode.steps))
            train_act.extend(map(lambda step: [step.action], episode.steps))
            no_graphs = False       # There is at least one graph to train on
    
    # We need the data to be in torch tensors for training
    train_obs_tensor = torch.tensor(train_obs, dtype = torch.float)
    train_act_tensor = torch.tensor(train_act, dtype = torch.float)

    return train_obs_tensor, train_act_tensor, reward_max, super_min_reward, batch_empty, no_graphs


if __name__ == '__main__':
    G = GraphEnv(conj21_rew_fn)
    net = Net(2*G.m, 1)
    objective = nn.BCEWithLogitsLoss()      # This builds in sigmoid when training... more stable than BCELoss()
    optimizer = optim.Adam(params = net.parameters(), lr = LEARNING_RATE) 
    writer = SummaryWriter()

    # Making a directory to save our plots to
    now = datetime.datetime.now()
    dir_name = "plots/plots" + now.strftime("%m-%d-%y_%H%M")
    os.makedirs(dir_name)
    
    # Variables for random actions when stuck in local minima:
    rand_act_prob = [RAND_ACT_PROB_INIT] # Using a list so that when pass to the generator, it can 
    # observe the changes we make to its content. That is, once a generator is instantiated, 
    # you can not change its inputs between yields.
    current_max_reward = -INF
    current_reward_start = 0

    frbddn_rwd = -INF   # Idea: when algorithm has converged to a non-counterexample, we want to
    # break out of that local minima, so we don't train on any of the graphs with that reward

    super_episodes = []     # Where we will keep our best epsides, and train on them until better are found
    super_min_reward = -INF     # Smallest reward of any super episode

    counterX_found = False  # We will print first counterexample found

    for iter_no, batch in enumerate(iterate_batches(G, net, rand_act_prob, BATCH_SIZE)):
        #observations, actions, max_rew, best_graph, is_empty = elites_from_batch(batch, PERCENTILE, frbddn_rwd) 
        observations, actions, max_rew, super_min_reward, batch_empty, no_graphs = \
            get_training_batch(batch, PERCENTILE, frbddn_rwd, super_episodes, SUPER_EPISODES, super_min_reward)

        if no_graphs:
            print("%d: \t No graphs satisfactory for training" % (iter_no))
            rand_act_prob[0] = min(RAND_ACT_PROB_MAX, rand_act_prob[0]*RAND_ACT_PROB_MULT)  # Need more random actions!
            continue # All graphs were disconnected or had forbidden reward, nothing to train on...
        
        optimizer.zero_grad()
        action_scores = net(observations)
        loss = objective(action_scores, actions)
        loss.backward()
        optimizer.step()

        print("%d: \t RAP: %.3f \t SMR: %.4f \t Loss: %.4f \t Max Reward: %.4f" % (iter_no, rand_act_prob[0], super_min_reward, loss.item(), max_rew))

        # Tweaking random action probability:
        if abs(current_max_reward - max_rew) > 0.0001:     # Reward changed, so decrease R.A.P.
            rand_act_prob[0] = max(RAND_ACT_PROB_INIT, rand_act_prob[0]/RAND_ACT_PROB_DVSR)
            current_max_reward = max_rew
            current_reward_start = iter_no
        elif iter_no - current_reward_start >= RAND_ACT_PROB_WAIT:  # Reward not changing, increase R.A.P.
            rand_act_prob[0] = RAND_ACT_PROB_MAX    # Just maximize RAP if in a local minima
            current_reward_start = iter_no
            frbddn_rwd = max_rew
            print(f"Forbidden Reward Update: {frbddn_rwd: .4f}")
        elif batch_empty:
            # Want more random actions if batch isn't giving anything to train on:
            rand_act_prob[0] = min(RAND_ACT_PROB_MAX, rand_act_prob[0]*RAND_ACT_PROB_MULT)

        # Monitoring the program:
        if iter_no % 3 == 0:
            writer.add_scalar("Loss", loss.item(), iter_no)
            writer.add_scalar("Max Reward", max_rew, iter_no)
        if iter_no % PLOT_RATE == 0:  # We will plot some graphs to visualize learning
            file_name = dir_name + f"/Iteration{iter_no}.png"
            plot_graph(super_episodes[-1].result.tolist()[0], file_name)
        if max_rew > 0 and not counterX_found:     # We will want to plot a counterexample as soon as it's found
            counterX_found = True
            file_name = dir_name + f"CounterExample{iter_no}.png"
            plot_graph(super_episodes[-1].result.tolist()[0], file_name)

        # Our exit case:
        if iter_no == 100_000:
            writer.close()
            break
