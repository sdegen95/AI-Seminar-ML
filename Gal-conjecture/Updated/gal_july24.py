import networkx as nx
import random
import numpy as np
from statistics import mean
import math
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import itertools

from numpy.linalg import matrix_power
from collections import Counter
#from cached_method import cached_method
import time

#from methodtools import lru_cache

# from multiprocessing import Pool
torch.set_num_threads(10)

# internal imports
# from combinatorics import reward

GAMES_PER_EPOCH    = 1000
TOP_GAMES_FACTOR   = 0.1
BATCH_SIZE         = 500
NETWORK_SIZES      = [1000, 200, 10]
LEARNING_RATE      = 0.001
EXPLORATION_RATE   = 0.05

NUMBER_VERTICES    = 20
DIMENSION = 5
CORRECT_EULER_CHAR = 0
if DIMENSION % 2 == 0:
    CORRECT_EULER_CHAR = 2
PUNISHMENT         = 100000000

class GalExample:
    def __init__(self,
                 n_vertices        = NUMBER_VERTICES, 
                 dim               = DIMENSION,
                 correct_ec        = CORRECT_EULER_CHAR,
                 punishment        = PUNISHMENT,
                 games_per_epoch   = GAMES_PER_EPOCH,
                 top_games_factor  = TOP_GAMES_FACTOR,
                 batch_size        = BATCH_SIZE,
                 network_sizes     = NETWORK_SIZES,
                 learning_rate     = LEARNING_RATE,
                 exploration_rate  = EXPLORATION_RATE
                ):
        # store input data
        self.n_vertices  = n_vertices
        self.n_edges     = n_vertices*(n_vertices-1)//2
        self.output_size = 1
        self.input_size  = self.n_edges
        self.dim         = dim
        self.punishment  = punishment
        self.correct_ec  = correct_ec

        # global variables
        self.games_per_epoch  = games_per_epoch
        self.top_games        = top_games_factor*games_per_epoch
        self.network_sizes    = network_sizes
        self.learning_rate    = learning_rate
        self.exploration_rate = exploration_rate
        self.batch_size       = batch_size

        # set model
        self.model     = None
        self.optimiser = None
        # The scheduler (currently unused)
        self.scheduler = None

        # computed top games
        self.found_top_games   = None
        self.ran_epochs        = 0
        self.top_games_counter = {}

    def run(self, epochs=50, restart=False, reset_model=False, use_scheduler=True):
        dim             = self.dim
        n_vertices      = self.n_vertices
        input_size      = self.input_size
        output_size     = self.output_size
        games_per_epoch = self.games_per_epoch

        if restart:
            self.found_top_games = None
            self.ran_epochs      = 0
        if self.found_top_games is None:
            restart = True
            board = np.zeros((games_per_epoch, input_size), dtype=np.float32)
            #for i in range(games_per_epoch):
            #    board[i, np.random.choice(np.arange(input_size), size=Hnr-dim)] = 1.
            self.found_top_games = board

        if reset_model:
            self.model           = None
            self.optimiser       = None
        if self.model is None:
            reset_model = True
            self.model = MatrixNeuralNetwork(self.network_sizes, input_size=input_size, output_size=output_size)
            self.optimiser = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            # Scheduler reduces the learning rate by a factor of 0.1 every step_size epochs
            self.scheduler = StepLR(self.optimiser, step_size=10, gamma=0.1)

        print(f"Running the algorithm (restart={ restart }, reset_model={ reset_model }) in dim={ self.dim} for { self.n_vertices } vertices with parameters:")
        print(f"    Games per epoch                         : { self.games_per_epoch }")
        print(f"    Batch size                              : { self.batch_size }")
        print(f"    Top games size                          : { int(self.top_games) }")
        print(f"    Exploration / Learning rate             : { self.exploration_rate } / { self.learning_rate }")
        print(f"    Neural network dimensions               : input={ input_size }, internal={self.network_sizes}")
        print()

        for i in range(epochs):
            new_games = self.play_games()
            # Mix these games into our pot of the best games so far.
            games = np.concatenate([self.found_top_games, new_games], axis=0)
            # Score the games, and rearrange into descending order by score.
            scores = self.score_fn(games)

            #scores = np.array(scores).astype(int)
            order = np.argsort(scores)[:int(self.top_games)]
            #Is = self.score_postprocessing([ self.get_inds(b) for b in games ], Rs)
            self.found_top_games = games[order]

            # training games
            self.reinforce_games(self.found_top_games)
            # updating scheduler
            if use_scheduler:
                self.scheduler.step()

            # Display progress
            self.ran_epochs += 1
            print(f"Epoch {self.ran_epochs} scores for { len(self.top_games_counter) } top games: {list(self.score_fn(self.found_top_games[:10]))}")
            if i % 10 == 0:
                top_adj = board_to_adj(self.n_vertices, self.found_top_games[:3])
                print("-------------------------")
                for game in top_adj:
                    fvec   = f_vector(game)
                    hvec   = h_vector(fvec)
                    gamvec = gamma_vector(hvec)
                    print(f"f-vec   : {fvec}")
                    print(f"h-vec   : {hvec}")
                    print(f"gam-vec : {gamvec}")

    def play_games(self):
        """
        Play games_per_epoch using the agent to give probability distributions on each move.
        """
        games_per_epoch = self.games_per_epoch
        input_size      = self.input_size
        output_size     = self.output_size
        dim             = self.dim
        board           = np.zeros((games_per_epoch, input_size), dtype=np.float32)
        moveid          = torch.eye(int(input_size))

        model           = self.model

        for step in range(input_size):
            # This ensures at least clique of large enough size
            if step >= input_size - dim*(dim-1)//2:
                board[:, step] = np.ones(games_per_epoch)
            exploration = (np.random.rand() < self.exploration_rate)
            if exploration:
                # Get random probabilities for exploration
                prob = np.random.rand(games_per_epoch)
                prob = prob / prob.sum()
            else:
                with torch.no_grad():
                    # Get probabilities from the model's predictions for exploitation
                    prob = model(torch.from_numpy(board), moveid[step:step+1, :]).view(games_per_epoch).numpy()

            # Set board values based on random values
            board[:, step] = np.random.rand(games_per_epoch) < prob
        return board

    def reinforce_games(self, games):
        """
        Given a list of completed games, reinforce each move in each game.
        """
        batch_size      = self.batch_size
        input_size      = self.input_size
        output_size     = self.output_size
        moveid          = torch.eye(int(input_size))

        model           = self.model
        optimiser       = self.optimiser

        n_games,_       = games.shape

        # Unpack the games into (state, move, actions).
        states  = torch.zeros((n_games, input_size, input_size))
        moveids = torch.zeros((n_games, input_size, input_size))
        actions = torch.zeros((n_games, input_size, output_size))
        for i in range(input_size):
            moveids[:, i, i] = 1
            states[:, i+1:, i] = torch.from_numpy(games[:, None, i])
            actions[:, i, 0] = torch.from_numpy(games[:, i])

        # Reshape these so that we can shuffle the moves between games.
        states  = states.flatten(end_dim=1)
        moveids = moveids.flatten(end_dim=1)
        actions = actions.flatten(end_dim=1)

        # Reinforce
        criterion = nn.BCELoss()
        # criterion = nn.CrossEntropyLoss()
        shuffle = torch.randperm(int(n_games * input_size))

        for i in range(0, n_games * input_size, batch_size):
            batch = shuffle[i:i+batch_size]

            optimiser.zero_grad()
            predicted = model(states[batch], moveids[batch])
            loss = criterion(predicted, actions[batch])
            loss.backward()
            optimiser.step()



    def score_fn(self, games):
        """
        I use the method reward instead of calling RewardFunction immediately
        so that I can cash the results and to not calculate the reward again.
        This becomes relevant only later when we get to the later checks in RewardFunction.
        """
        """
        scores = []
        for game in games:
            #adj = self.board_to_adj(game)
            score = int(self.reward(game))
            #print(game)
            #score = 5
            scores.append(score)
        return scores
        """
        return [ int(self.reward(game)) for game in games ]

    
    def reward(self, adj):
        return RewardFunction(adj, self.n_vertices, self.dim, self.punishment, self.correct_ec)

    def score_postprocessing(self, As, Rs):
        Rs, As, Is = list(zip(*sorted(zip(Rs,As,range(len(As))), reverse=True)))
        Rs = list(Rs)
        top_games = int(self.top_games)
        for i,A in enumerate(As):
            if i < top_games:
                l = self.top_games_counter.get(A, 0)
                self.top_games_counter[A] = l+1
                #Rs[i] -= l
            A = set(A)
            if 0 < i < 2*top_games:
                min_dist = min([len(A) - len(A.intersection(As[j])) for j in range(i)])
                if min_dist > 2:
                    Rs[i] += (min_dist-2)

        RAIs = sorted(zip(Rs,As,Is), reverse=True)

        seen = set()
        Is = [i for _,A,i in RAIs if not (A in seen or seen.add(A))]

        return Is[:int(self.top_games)]

 

class MatrixNeuralNetwork(nn.Module):
    def __init__(self, layer_dims, input_size, output_size):
        super().__init__()

        M = input_size
        # Deal with the state and one-hot move vectors separately, since we can do this efficiently.
        self.first_state = nn.Linear(M, layer_dims[0])
        self.first_move = nn.Linear(M, layer_dims[0], bias=False)

        self.layers = nn.ModuleList([
            nn.Linear(a, b)
            for a, b in zip(layer_dims, layer_dims[1:])
        ])

        self.fully_conn = nn.Linear(layer_dims[-1], output_size)

    def forward(self, state, move):
        x = self.first_state(state) + self.first_move(move)
        x = nn.functional.relu(x)

        for linear in self.layers:
            x = linear(x)
            x = nn.functional.relu(x)

        x = self.fully_conn(x)
        # val = torch.softmax(x, dim=1)
        val = torch.sigmoid(x)
        return val

#@lru_cache
def RewardFunction(board, n_vertices, dim, punishment, correct_ec):
    # Check if graph is connected
    #conn = Main.check_connected(adj)
    adj = board_to_adj(n_vertices, board)

    if np.sum(adj) < n_vertices: #or np.sum(adj) > 26:
        return punishment * n_vertices
    
    G = nx.convert_matrix.from_numpy_array(adj)
    # if nx.node_connectivity(G) == 0:
    #     return int(punish_value/2)
    # if nx.node_connectivity(G) == 1:
    #     return int(punish_value/4)
    # if nx.node_connectivity(G) == 2:
    #     return int(punish_value/5)

    # The graph of a flag manifold must be 2d connected
    conn = nx.node_connectivity(G)
    if conn < 2 * dim:
        return ((2 * dim) - conn) * punishment

    #f_vec, cliques = f_vector(adj)
    f_vec = f_vector(adj)

    if len(f_vec) != dim+2:
        return  int(punishment/5)

    if euler_char(f_vec) != correct_ec:
        return int(punishment/5)

    # minimal values based on paper "Some combintorial properties of flag simplicial pseudomanifolds and spheres"
    min_fvec = [2^i*math.comb(dim+1,i) for i in range(dim+2)]
    diff_fvec = sum([max(min_fvec[i]-f_vec[i],0) for i in range(dim+2)])
    if diff_fvec > 0:
        return diff_fvec*punishment/100

    h_vec = h_vector(f_vec)
    
    min_hvec = [math.comb(dim+1,i) for i in range(dim+2)]
    diff_hvec = sum([max(min_hvec[i]-h_vec[i],0) for i in range(dim+2)])
    if diff_hvec > 0:
        return diff_hvec*punishment/1000
    
    """
    d = len(h_vec)-1
    neg_components = abs(sum([h for h in h_vec if h < 0]))

    if neg_components > 0:
        return neg_components*punishment/10000
    """

    #if d+1 == dim+2:
    t = sum([abs(h_vec[d-i]-h_vec[i]) for i in range(d//2+1)])
    if t > 0:
        return 100*t

    gam_vec = gamma_vector(h_vec)
    gam_min = min(gam_vec)
    #return gam_min

    if gam_min < 0:
        return 0
    return gam_min

"""
if gam_min >= LOWER_BOUND: 
    considered_cliques = [x for x in cliques if len(x)==d-1 or len(x)==d-2]
    for face in considered_cliques:
        l = link(cliques,face)
        l_h_vec = h_vector_from_simplices(l)
            l_d = len(l_h_vec)-1
            l_t = sum([abs(l_h_vec[l_d-i]-l_h_vec[i]) for i in range(l_d//2+1)])
            if l_t > 0:
                gam_min += l_t*0.01
            elif l_t == 0:
                continue
        return gam_min
    else:
        return abs(gam_min)
else:
    return gam_min + 100
else:
    return neg_components + 10000

"""

# ## Helper functions

def board_to_adj(N, board):
    """
    Input: A tensor of shape (*, M), giving a (batched) 01-sequence of edges.
    Output: A tensor of shape (*, N, N), giving a (batched) adjacency matrix.
    """
    adj = np.zeros([*board.shape[:-1], N, N], dtype=np.int8)
    count = 0
    for i in range(N):
        for j in range(i+1, N):
            adj[..., i, j] = adj[..., j, i] = board[..., count] != 0
            count += 1

    return adj

def display_adj(adj):
    #print(adj)
    G = nx.convert_matrix.from_numpy_array(adj)
    plt.clf()
    plt.axis('equal')
    nx.draw_circular(nx.complement(G))
    plt.savefig("mygraph.png")

#@lru_cache
def f_vector(adj):
    G = nx.convert_matrix.from_numpy_array(adj)
    cliques = list(nx.enumerate_all_cliques(G))
    faces_count = Counter([len(elm) for elm in cliques])
    f_vec = [faces_count[i] for i in range(len(faces_count)+1)]
    f_vec[0] = 1
    return f_vec#, cliques

def h_vector(f_vec):
    h_vec = []
    d = len(f_vec)-1

    for k in range(d+1):
        h_k = 0
        for i in range(k+1):
            h_k += (-1)**(k-i)*math.comb(d-i,k-i)*f_vec[i]
        h_vec.append(h_k)
    
    return h_vec

def is_symmetric(h_vec):
    d = len(h_vec)
    split_index = (len(h_vec)-1)//2
    is_symm = True
    
    if d % 2 != 0:
        for i in range(split_index):
            if h_vec[i] == h_vec[-(i+1)]:
                continue
            else:
                is_symm = False
                break
    elif d % 2 == 0:
        for i in range(split_index+1):
            if h_vec[i] == h_vec[-(i+1)]:
                continue
            else:
                is_symm = False
                break
    
    return is_symm

def gamma_vector(h_vec):
    gam_vec = []
    n = len(h_vec)-1
    d = n//2

    # gamma 0
    gam_vec.append(h_vec[0])

    # gamma 1 to d
    for i in range(1,d+1):
        g_k = h_vec[i]
        for j in range(i):
            g_k -= math.comb(n-2*j,i-j)*gam_vec[j]
        gam_vec.append(g_k)

    return gam_vec

#@cached_method
def euler_char(f_vec):
    e_char = 0
    new_f_vec = [f_vec[i] for i in range(len(f_vec)) if i != 0]

    # Sum up the entry of the new_f_vec in an alternating manner
    for i in range(len(new_f_vec)):
        if i % 2 == 0:
            e_char += new_f_vec[i]
        else:
            e_char -= new_f_vec[i]
    
    return e_char
