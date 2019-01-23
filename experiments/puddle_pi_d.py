#!/usr/bin/env python

# Python imports.
import sys
import numpy as np

# Other imports.
import puddle_info as pi
from simple_rl.agents import LinearQAgent, QLearningAgent, FixedPolicyAgent, Agent
from simple_rl.tasks import PuddleMDP
from simple_rl.run_experiments import run_agents_on_mdp
from simple_rl.run_experiments import evaluate_agent

GAP = 0.07

def get_demo_policy_given_goal(goal_loc):
    '''
    Args:
        goal_loc (list)

    Returns:
        (lambda): One of the below demo policies for puddle.
    '''
    if goal_loc == [1.0, 1.0]:
        # Top Right.
        return expert_puddle_policy_top_right
    elif goal_loc == [0.0, 1.0]:
        # Top Left.
        return expert_puddle_policy_top_left
    elif goal_loc == [0.0, 0.0]:
        # Bottom left.
        return expert_puddle_policy_bot_left
    elif goal_loc == [1.0, 0.0]:
        # Bottom right.
        return expert_puddle_policy_bot_right
    else:
        raise ValueError("(em_rl_abstraction): Unknown goal `" + str(goal_loc) + "'. Accepted values are : [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0].")

def expert_puddle_policy_top_right(state):
    '''
    Args:
        state (simple_rl.State)

    Returns:
        (str)

    Summary:
        An "expert" policy to solve the traditional puddle world game.
    '''

    if (state.x <= pi.pud_out_right_x and state.y < pi.pud_out_bottom_y): # or (state.x > 0.5 and state.y >= 0.95) or (state.y >= 0.8):
        # Bottom row.
        action = "right"
    elif state.x >= pi.pud_out_right_x and state.y <= 1.0 - GAP:
        # Right column.
        action = "up"
    elif state.x >= pi.pud_out_right_x and state.y >= 1.0 - GAP:
        # Top right.
        action = "right"
    elif state.x <= pi.pud_out_left_x and state.y <= pi.pud_out_top_y:
        # Left column.
        action = "up"
    elif state.y >= pi.pud_out_top_y:
        # Top row.
        action = "right"
    elif state.x <= pi.pud_in_left_x and state.y <= pi.pud_in_bottom_y:
        # Beneath/to left of puddle.
        action = "down"
    else:
        action = "left"
    return action

def expert_puddle_policy_top_left(state):
    '''
    Args:
        state (simple_rl.State)

    Returns:
        (str)

    Summary:
        An "expert" policy to solve the traditional puddle world game, with the goal in the top left.
    '''
    # pud_out_left_x
    # pud_out_top_y
    # pud_in_bottom_y
    # pud_out_right_x
    # pud_out_bottom_y
    # pud_in_left_x
    if state.x <= GAP:
        # Left column.
        action = "up"
    elif state.y >= pi.pud_out_top_y:
        # Top row.
        action = "left"
    elif state.x >= pi.pud_out_right_x:
        # Right of puddles.
        action = "up"
    elif state.y <= pi.pud_out_bottom_y:
        # Bottom of puddles.
        action = "left"
    elif state.y <= pi.pud_in_bottom_y and state.x <= pi.pud_in_left_x:
        # Starting region (nestled by both puddles).
        action = "left"
    elif state.x >= GAP:
        # Top row.
        action = "left"
    else:
        action = "left"
    return action

def expert_puddle_policy_bot_left(state):
    '''
    Args:
        state (simple_rl.State)

    Returns:
        (str)

    Summary:
        An "expert" policy to solve the traditional puddle world game, with the goal in the top left.
    '''
    if state.y <= GAP:
        # Bottom row.
        action = "left"
    if state.x <= GAP:
        # Left column.
        action = "down"
    elif state.y >= pi.pud_out_top_y:
        # Top row.
        action = "left"
    elif state.x >= pi.pud_out_right_x:
        # Right of puddles.
        action = "down"
    elif state.y <= pi.pud_out_bottom_y:
        # Bottom of puddles.
        action = "left"
    elif state.y <= pi.pud_in_bottom_y and state.x <= pi.pud_in_left_x:
        # Starting region (nestled by both puddles).
        action = "down"
    elif state.x >= GAP:
        # Top row.
        action = "left"
    else:
        action = "down"
    return action

    # if state.x >= pi.pud_out_right_x and state.y >= GAP:
    #     # To the right of the puddle.
    #     action = "down" 
    # elif state.x <= GAP:
    #     # Left column.
    #     action = "down"
    # elif state.x >= GAP:
    #     # Right column.
    #     action = "left"
    # return action

def expert_puddle_policy_bot_right(state):
    '''
    Args:
        state (simple_rl.State)

    Returns:
        (str)

    Summary:
        An "expert" policy to solve the traditional puddle world game, with the goal in the top left.
    '''
    if state.y <= GAP:
        # Bottom row.
        action = "right"
    elif state.y >= pi.pud_out_top_y and state.x < 1.0 - GAP:
        # Top row.
        action = "right"
    elif state.y >= GAP:
        # Right column and left column.
        action = "down"
    return action

def stochastic_expert_policy(state, beta=1.0):
    '''
    Args:
        state (simple_rl.State)
        beta (float)

    Returns:
        (str)
    '''

    q_vals = []
    for a in actions:
        # Get Q Value for that action.
        next_agent = ActThenPiAgent(a, expert_puddle_policy)
        q_vals.append(evaluate_agent(next_agent, mdp))

    # Softmax distribution.
    total = sum([np.exp(beta * qv) for qv in q_vals])
    softmax = [np.exp(beta * qv) / total for qv in q_vals]

    return np.random.choice(actions, 1, p=softmax)[0]
'''
def main(open_plot=True):

    # Make MDP.
    from run_learning_experiment import make_mdp_demo_policy_dict

    mdp_demo_policy_dict, test_mdp = make_mdp_demo_policy_dict(multitask=True)
    expert_puddle_policy = get_demo_policy_given_goal(test_mdp.get_goal_locs()[0])
    demo_agent = FixedPolicyAgent(expert_puddle_policy)

    print "Goals:", test_mdp.get_goal_locs()

    # Run experiment and make plot.
    run_agents_on_mdp([demo_agent], test_mdp, instances=3, episodes=1, steps=2000, open_plot=open_plot)

if __name__ == "__main__":
    main()
'''
'''
import numpy
from simple_rl.tasks import PuddleMDP
import matplotlib.pyplot as plt
actions={}
actions['right']=0 #red
actions['up']=1 #blue
actions['left']=2 #black
actions['down']=3 #green
colors=['red','blue','black','green']

x_li=[]
y_li=[]
c_li=[]
for x in numpy.arange(0,1,.01):
    for y in numpy.arange(0,1,.01):
        state=PuddleMDP().get_init_state()
        state.x=x
        state.y=y
        x_li.append(x)
        y_li.append(y)
        #a=expert_puddle_policy_bot_left(state) #needs to be revised
        #a=expert_puddle_policy_top_left(state) #needs to be revised
        #a=expert_puddle_policy_bot_right(state) #looks good
        a=expert_puddle_policy_top_right(state) #looks good
        index=actions[a]
        c_li.append(colors[index])
plt.scatter(x_li,y_li,c=c_li)
plt.show()
plt.close()
'''

