# Python imports.
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D

# simple_rl imports.
from simple_rl.agents import RandomAgent

color_ls = [[169, 193, 213], [230, 169, 132],\
            [198, 113, 113],[192, 197, 182], [210, 180, 226],[94, 94, 94],\
            [118, 167, 125], [102, 120, 173],]

color_ls = [[shade / 255.0 for shade in rgb] for rgb in color_ls]

def collect_dataset(mdp, samples=10000, learning_agent=None):
    '''
    Args:
        mdp (simple_rl.MDP)
        samples (int)
        learning_agent (simple_rl.Agent): If None, a random agent is used.
            Otherwise collects data based on its learning.

    Returns:
        (set)
    '''
    if learning_agent is None:
        learning_agent = RandomAgent(mdp.get_actions())

    cur_state = mdp.get_init_state()
    reward = 0
    visited_states = set([cur_state])
    
    # Set initial state params.
    init_state_params = {}
    last_x = 0 + np.random.randn(1)[0]
    init_state_params["x"] = last_x
    init_state_params["x_dot"] = 0
    init_state_params["theta"] = 0
    init_state_params["theta_dot"] = 0


    for i in range(samples):
        action = learning_agent.act(cur_state, reward)
        reward, next_state = mdp.execute_agent_action(action)

        visited_states.add(next_state)
        if next_state.is_terminal():
            init_state_params["x"] = np.random.randn(1)[0]
            mdp.reset(init_state_params)
            learning_agent.end_of_episode()
            cur_state = mdp.get_init_state()
            reward = 0
        else:
            cur_state = next_state

    return visited_states



def plot_state_abstr(feat_a, feat_b, dataset, nn_sa):
    '''
    Args:
        feat_a (dict): Contains a "name" and "index" key.
        feat_b (dict): Contains a "name" and "index" key.
        a_range (generator): see func @drange below)
        b_range (generator): see func @drange below)
        dataset (set): Contains visited states to visualize
        nn_sa
    '''
    # Get feature information.
    feat_a_name, feat_a_index = feat_a["name"], feat_a["index"]
    feat_b_name, feat_b_index = feat_b["name"], feat_b["index"]

    # Make dataset of (feat_a_val, feat_b_val, color)
    x, y,  = np.zeros(len(dataset)), np.zeros(len(dataset))
    colors = [0] * len(dataset)
    for i, state in enumerate(dataset):
        feat_a, feat_b = state[feat_a_index], state[feat_b_index]
        cluster = nn_sa.phi(state)
        x[i], y[i] = feat_a, feat_b
        colors[i] = tuple(color_ls[cluster.data])

    # Plot
    colors = np.array(colors)
    for x, y, c in zip(x, y, colors):
        c = tuple(c)
        plt.scatter(x, y, c=c)
    plt.title(feat_a_name + " vs. " + feat_b_name)
    plt.xlabel(feat_a_name)
    plt.ylabel(feat_b_name)
    # plt.show()

    plt.savefig(os.path.join("visuals", feat_a_name.lower().replace(" ", "_") + "_vs_" + feat_b_name.lower().replace(" ", "_")) + ".pdf", format="pdf")
    plt.cla()
    plt.close()

def get_feature_pairs(features):
    '''
    Args:
        features (list)

    Returns:
        (list)
    '''
    
    pairs = []
    for pair in itertools.product(features, repeat=2):
        pair = list(pair)
        pair.sort()

        if pair not in pairs and pair[0] is not pair[1]:
            pairs.append(pair)

    return pairs


# =======================
# == Main Viz Function ==
# =======================

def visualize_state_abstrs(state_dataset, features, nn_sa):
    '''
    Args:
        state_dataset (list)
        features (dict)
        nn_sa (NNStateAbstr)

    Summary:
        Makes all pairwise feature plots of the nn_sa.
    '''
    # Make unique feature pairs to plot.
    feature_pairs = get_feature_pairs(features)

    # Make plots.
    for feat_a, feat_b in feature_pairs:
        plot_state_abstr(feat_a, feat_b, state_dataset, nn_sa)


def visualize_state_abstrs3D(state_dataset, features, nn_sa):
    state_li=[list(state[0]) for state in state_dataset]
    phi_li=[nn_sa.phi(state).data for state in state_li]
    
    plot3D(state_li,phi_li,features)

def plot3D(state_li,phi_li,features):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_dot_li = [state[1] for state in state_li]
    theta_li = [state[2] for state in state_li]
    theta_dot_li = [state[3] for state in state_li]
    color_li = [color_ls[c] for c in phi_li]
        
    ax.scatter(x_dot_li, theta_li, theta_dot_li, color=color_li)

    ax.set_xlabel('Velocity')
    ax.set_ylabel('Angle')
    ax.set_zlabel('Angular Velocity')

    ax.set_xticks([-1,0,1])
    ax.set_yticks([-.1,0,.1])
    ax.set_zticks([-1,0,1])
    
    plt.show()    
    plt.close()
