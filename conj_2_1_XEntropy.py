'''
In AZ Wagner's paper 'Constructions in combinatorics via nerual networks,' he uses the 
cross-entropy RL method to find a 19-vertex counterexample to the following:

Conjecture 2.1: Let G be a connected graph on n >= 3 vertices, with largest 
eigenvalue \lambda_1 and matching number \mu. Then \lambda_1 + \mu >= \sqrt{n-1} + 1.

This is my own code written to recreate his result. There can be a large difference run-to-run, 
sometimes converging to a non-counterexample. With the given hyperparameters, it did
converge to the intended counterexample in less than 200 iterations on several runs,
each time taking less than ten minutes on my 2020 MacBook Pro. 
'''
from collections import namedtuple  # Helps us write cleaner code
from tensorboardX import SummaryWriter     # To help visualize our runs
import matplotlib.pyplot as plt     # For drawing our graphs
import networkx as nx   # Graph package
import numpy as np      # Helpful for some computations
import math             # Helpful for other computations
# PyTorch for deep learning:
import torch
import torch.nn as nn
import torch.optim as optim

N = 19                  # Number of vertices
HIDDEN_SIZE = 128       # The NN has two layers with HIDDEN_SIZE number of neurons
BATCH_SIZE = 100        # Number of graphs 'played' in each batch
PERCENTILE = 90         # Only the top (100 - PERCENTILE)% graphs in each bartch are used to train the NN
LEARNING_RATE = 0.006   # Was not converging to the desired counterexample when > 0.01
INF = 2**15             # Placeholder

class GraphEnv():
    ''' A graph environment that utilizes the one-hot encoding method used by AZ Wagner
    -- intended to partially emulate a gymnasium RL environment '''
    def __init__(self,  reward_fn, num_verts = N):
        self.n = num_verts
        self.m = int((self.n * (self.n - 1))/2)
        self.state = torch.zeros((2, self.m), dtype = torch.float)
        self.state[1,0] = 1     # Our one-hot bit
        self.terminated = False
        self.rew = 0.0
        self.reward_function = reward_fn    # We can pass any reward function to this environment
        self.cur_edge = 0                   # 'Follows' the one-hot bit

    def reset(self):
        self.state.zero_()
        self.state[1,0] = 1 
        self.terminated = False
        self.cur_edge = 0
        self.rew = 0.0

        return self.state.reshape(1, 2*self.m)  # We reshape state to what the NN wants to see

    def step(self, action):
        if not self.terminated:
            self.state[0,self.cur_edge] = action
            self.state[1,self.cur_edge] = 0
            # For X-Entropy, the following need only be called after episode terminated
            # self.reward += self.reward_function(self.state)
            self.cur_edge += 1

            if self.cur_edge >= self.m: # Whole graph determined
                self.terminated = True
                # We only update the reward once the whole graph is determined b/c X-entropy only
                # needs the reward to select the elite episodes from the batch
                self.rew = self.reward_function(self.state, self.n)
            else:
                self.state[1, self.cur_edge] = 1    # Push one-hot bit along
        
        # Gym environments return two other extra info lists, we don't...
        # Other note: we reshape state below to what the NN wants to see
        return self.state.reshape(1, 2*self.m), self.rew, self.terminated


def conj21_rew_fn(state, num_verts):
    ''' The reward function for Conjecture 2.1, see Wagner paper  '''
    G = nx.Graph()
    G.add_nodes_from(list(range(num_verts)))
    count = 0

    for i in range(num_verts):
        for j in range(i+1, num_verts):
            if state[0, count] == 1:
                G.add_edge(i,j)
            count += 1
    
    if not nx.is_connected(G):
        return -INF
    
    evals = np.linalg.eigvalsh(nx.adjacency_matrix(G).todense())
    evalsRealAbs = np.zeros_like(evals)

    for i in range(len(evals)):
        evalsRealAbs[i] = abs(evals[i])
    lambda1 = max(evalsRealAbs)

    maxMatch = nx.max_weight_matching(G)
    mu = len(maxMatch)

    score = math.sqrt(num_verts-1) + 1 - lambda1 - mu # Conjecture broken iff score > 0

    not_drawn = True
    if score > 0 and not_drawn:  
        # Good place for this since the nx graph is already made!
        print('Counterexample Found!!!', state[0])
        nx.draw_kamada_kawai(G)
        plt.show(block=False)   # block=False so code keeps running
        not_drawn = False       # We only want to draw one graph...

    return score

        
class Net(nn.Module):
    ''' Standard creation of a NN with PyTorch '''
    def __init__(self, in_size, out_size, layer_1 = HIDDEN_SIZE, layer_2 = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, layer_1),
            nn.ReLU(),
            nn.Linear(layer_1, layer_2),
            nn.ReLU(),
            nn.Linear(layer_2, out_size)
        )

    def forward(self, x):
        return self.net(x)
    
# The following are used to write cleaner code --  they return new tuple subclasses
Episode = namedtuple('Episode', field_names=['reward', 'result', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

def iterate_batches(env, net, batch_size = BATCH_SIZE):
    ''' Generates batches of graphs '''
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sig = nn.Sigmoid()      # Turns outputs into probabilities, (range is (0,1))

    while  True:
        act_probs = sig(net(obs))
        
        if np.random.rand() < act_probs:
            action = 1      # The larger act_probs, the more likely to add the edge
        else:
            action = 0

        step = EpisodeStep(observation = obs.clone(), action = action)
        next_obs, reward, terminated = env.step(action)
        episode_reward += reward
        episode_steps.append(step)

        if terminated:
            # Need obs.clone() below for aliasing reasons
            e = Episode(reward = episode_reward, result = obs.clone(), steps = episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = []

        obs = next_obs


def elites_from_batch(batch, percentile):
    ''' Filters the batch, returning the best episodes '''
    rewards = list(map(lambda s: s.reward, batch)) # Extracts all rewards from the batch
    reward_bound = np.percentile(rewards, percentile)  # The min reward required to use for training
    reward_max = float(np.max(rewards))

    train_obs = []
    train_act = []

    empty = True    # Possible no connected graphs in batch, so there is nothing to train on

    for reward, result, steps in batch:
        if reward < reward_bound or reward == -INF:
            continue
        train_obs.extend(map(lambda step: step.observation.tolist()[0], steps))
        train_act.extend(map(lambda step: [step.action], steps))
        empty = False 

    # We need the data to be torch tensors for training   
    return torch.tensor(train_obs, dtype = torch.float), torch.tensor(train_act, dtype = torch.float), reward_max, empty

if __name__ == '__main__':
    G = GraphEnv(conj21_rew_fn)
    net = Net(2*G.m, 1)
    objective = nn.BCEWithLogitsLoss()      # This builds in sigmoid when training... more stable than BCELoss()
    optimizer = optim.Adam(params = net.parameters(), lr = LEARNING_RATE)
    writer = SummaryWriter()

    for iter_no, batch in enumerate(iterate_batches(G, net, BATCH_SIZE)):
        observations, actions, rew_max, is_empty = elites_from_batch(batch, PERCENTILE) 

        if is_empty:
            print("%d: \t All graphs disconnected... :(" % (iter_no))
            continue # All graphs were disconnected, nothing to train on...
        
        optimizer.zero_grad()
        action_scores = net(observations)
        loss = objective(action_scores, actions)
        loss.backward()
        optimizer.step()

        print("%d:  loss= %.4f,  max reward= %.3f" % (iter_no, loss.item(), rew_max))
        writer.add_scalar("Loss", loss, iter_no)
        writer.add_scalar("Max Reward", rew_max, iter_no)

        # To watch it learn, we could plot the best graph every 50-100ish iterations...
        # We might want to have elites_from_batch() return the state of the best graph so we can do this here

        if rew_max > 0 or loss < 0.00001 or iter_no == 1_000:
            writer.close()
            break