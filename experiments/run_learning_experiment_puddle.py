# Python imports.
import sys
import random
import gym
import numpy
import os

# Other imports.
import tensorflow as tf
from simple_rl.abstraction.AbstractionWrapperClass import AbstractionWrapper
from simple_rl.agents import QLearningAgent, LinearQAgent, FixedPolicyAgent
from simple_rl.tasks import PuddleMDP, GymMDP
from simple_rl.run_experiments import run_agents_on_mdp, evaluate_agent
from simple_rl.utils import chart_utils as cu

# Local imports.
from NNStateAbstrClass import NNStateAbstr
import puddle_pi_d as ppd
import puddle_info as pi
from experiment_utils import make_nn_sa,plot_learned_abstraction

def get_params(default_params=None):
    '''
    Args:
        default_params (dict): May contain default settings for parameters.

    Summary:
        Defines parameters for use in experiments.
    '''
    params={}
    params['multitask']=True
    params['env_name']="PuddleMDP"
    params['obs_size']=2
    params['num_iterations_for_abstraction_learning']=1000
    params['learning_rate_for_abstraction_learning']=0.001
    params['abstraction_network_hidden_layers']=2
    params['abstraction_network_hidden_nodes']=200
    params['num_samples_from_demonstrator']=400
    params['episodes']=250 if params['multitask'] else 100
    params['steps']=1000
    params['num_instances']=25

    # Set defaults if given.
    if default_params is not None:
        for key in default_params.keys():
            params[key] = default_params[key]

    return params

def diff_sampling_distr_experiment():
    '''
    Summary:
        Compares performance of different sample styles to compute phi.
    '''
    # Make MDP and Demo Policy.
    params = get_params()
    mdp_demo_policy_dict, test_mdp = make_mdp_demo_policy_dict(multitask=False)
    expert_puddle_policy = ppd.get_demo_policy_given_goal(test_mdp.get_goal_locs()[0])
    demo_agent = FixedPolicyAgent(expert_puddle_policy)

    # Make a NN for each sampling param.
    agents = {}
    sess = tf.Session()
    sampling_params = [0.0, 0.5, 1.0]

    for epsilon in sampling_params:
        with tf.variable_scope('nn_sa' + str(epsilon), reuse=False) as scope:
        # tf.reset_default_graph()
            params["epsilon"] = epsilon
            abstraction_net = make_nn_sa(mdp_demo_policy_dict, sess, params, verbose=False, sample_type="demo")
            nn_sa = NNStateAbstr(abstraction_net)
            sa_agent = AbstractionWrapper(QLearningAgent, agent_params={"actions":test_mdp.get_actions(), "name":"$D \\sim \\rho_E^\\epsilon, \\epsilon=" + str(epsilon) + "$"}, state_abstr=nn_sa, name_ext="")
            agents[epsilon] = sa_agent

    with tf.variable_scope('demo') as scope:
        abstraction_net_rand = make_nn_sa(mdp_demo_policy_dict, sess, params, verbose=False, sample_type="rand")
        nn_sa_rand = NNStateAbstr(abstraction_net_rand)
        sa_agent_rand = AbstractionWrapper(QLearningAgent, agent_params={"actions":test_mdp.get_actions(), "name":"$D \\sim U(S)$"}, state_abstr=nn_sa_rand, name_ext="")
        agents["rand"] = sa_agent_rand

    run_agents_on_mdp(agents.values(), test_mdp, instances=params['num_instances'], episodes=params['episodes'], steps=params['steps'], verbose=False)

    sess.close()


def num_training_data_experiment():
    '''
    Summary:
        Runs an experiment that compares the performance of different
        Agent-SA combinations, where each SA is trained with a different
        number of training samples.
    '''
    # Params.
    instances = 10
    init, increment, maximum = 1, 500, 5001
    training_samples = range(init, maximum, increment)

    # Run experiment.s
    if not os.path.exists(os.path.join("results", "puddle_per_sample")):
        os.makedirs(os.path.join("results", "puddle_per_sample"))
    data_dir = os.path.join("results", "puddle_per_sample")
    with open(os.path.join(data_dir, "results.csv"), "w+") as results_file:
        
        # Repeat the experiment @instances # times.
        for i in range(instances):
            print "\nInstances", i + 1, "of", str(instances)
            for sample_num in training_samples:
                print "\tSamples:", sample_num

                # Make State Abstraction.
                params = get_params(default_params={"num_samples_from_demonstrator":sample_num})
                mdp_demo_policy_dict, test_mdp = make_mdp_demo_policy_dict(multitask=params['multitask'])
                expert_puddle_policy = ppd.get_demo_policy_given_goal(test_mdp.get_goal_locs()[0])
                demo_agent = FixedPolicyAgent(expert_puddle_policy)
                tf.reset_default_graph()
                sess = tf.Session()
                abstraction_net = make_nn_sa(mdp_demo_policy_dict, sess, params, verbose=False)
                nn_sa = NNStateAbstr(abstraction_net)

                # Test Performance with given param.
                sa_agent = AbstractionWrapper(QLearningAgent, agent_params={"actions":test_mdp.get_actions()}, state_abstr=nn_sa, name_ext="$-\\phi$")
                val = evaluate_agent(sa_agent, test_mdp, steps=params['steps'], episodes=params['episodes'])
                results_file.write(str(val) + ",")
                results_file.flush()
                sess.close()

            results_file.write("\n")

    cu.EVERY_OTHER_X = True
    cu.CUSTOM_TITLE = "Effect of $|D_{train, \\phi}|$ on RL Performance"
    cu.X_AXIS_LABEL = "$|D_{train, \\phi}|$"
    cu.Y_AXIS_LABEL = "Avg. Reward in Last Episode"
    cu.X_AXIS_START_VAL = init
    cu.X_AXIS_INCREMENT = increment
    cu.COLOR_SHIFT = 3
    cu.format_and_make_plot(data_dir=data_dir, avg_plot=True, add_legend=False)


def make_mdp_demo_policy_dict(multitask, task="puddle"):
    '''
    Args:
        multitask (bool)

    Returns:
        (dict)
        (simple_rl.MDP)
    '''
    step_cost = 0.001
    goal_locs = [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
    if not multitask:
        # Single task.
        if task == "puddle":
            list_of_mdps = [PuddleMDP(puddle_rects=pi.PUDDLE, step_cost=step_cost, goal_locs=[goal_locs[0]])]
        test_mdp = list_of_mdps[0]
    else:
        # Multitask.
        if task == "puddle":
            list_of_mdps = [PuddleMDP(puddle_rects=pi.PUDDLE, step_cost=step_cost, goal_locs=[goal_locs[i]]) for i in range(len(goal_locs))]
        else:
            raise ValueError("No such MDP: " + str(task))
        random.shuffle(list_of_mdps)
        test_mdp = list_of_mdps.pop()
        print "\nTest Goal:", test_mdp.get_goal_locs()[0], "\n"

    # Make mdp-policy dictionary.
    mdp_demo_policy_dict = {}
    for mdp in list_of_mdps:
        mdp_demo_policy_dict[mdp] = ppd.get_demo_policy_given_goal(mdp.get_goal_locs()[0])

    return mdp_demo_policy_dict, test_mdp


def main():

    # ======================
    # == Make Environment ==
    # ======================
    params = get_params()

    # ============================
    # == Make test and train environments
    # == along with demonstrator(s)
    # ============================
    mdp_demo_policy_dict, test_mdp = make_mdp_demo_policy_dict(multitask=params['multitask'])
    expert_puddle_policy = ppd.get_demo_policy_given_goal(test_mdp.get_goal_locs()[0])
    demo_agent = FixedPolicyAgent(expert_puddle_policy)

    # ============================
    # == Make State Abstraction ==
    # ============================
    sess = tf.Session()
    abstraction_net = make_nn_sa(mdp_demo_policy_dict, sess,params)
    nn_sa = NNStateAbstr(abstraction_net)

    # =================
    # == Make Agents ==
    # =================
    actions = test_mdp.get_actions()
    num_features = test_mdp.get_num_state_feats()
    linear_agent = LinearQAgent(actions=actions, num_features=num_features)
    sa_agent = AbstractionWrapper(QLearningAgent, agent_params={"actions":test_mdp.get_actions()}, state_abstr=nn_sa, name_ext="$-\\phi$")

    # ====================
    # == Run Experiment ==
    # ====================
    run_agents_on_mdp([sa_agent, linear_agent], test_mdp, instances=params['num_instances'],
                     episodes=params['episodes'], steps=params['steps'],
                     verbose=False)

if __name__ == "__main__":
    # main()
    num_training_data_experiment()
