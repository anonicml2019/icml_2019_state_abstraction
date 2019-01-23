# Python imports.
import sys
import random
import tensorflow as tf
import gym

# simple_rl imports.
from simple_rl.tasks import GymMDP
from simple_rl.abstraction.AbstractionWrapperClass import AbstractionWrapper
from simple_rl.agents import QLearningAgent, LinearQAgent, FixedPolicyAgent
from simple_rl.tasks import PuddleMDP
from simple_rl.run_experiments import run_agents_on_mdp, evaluate_agent

# Local imports.
from NNStateAbstrClass import NNStateAbstr
import lunar_pi_d as lpd
from experiment_utils import make_nn_sa

def main():

    # ======================
    # == Make Environment ==
    # ======================
    params={}
    params['multitask']=False
    params['env_name']="LunarLander-v2"
    params['obs_size']=8
    params['num_iterations_for_abstraction_learning']=500
    params['learning_rate_for_abstraction_learning']=0.005
    params['abstraction_network_hidden_layers']=2
    params['abstraction_network_hidden_nodes']=200
    params['num_samples_from_demonstrator']=10000
    params['episodes']=200
    params['steps']=1000
    params['num_instances']=5
    params['rl_learning_rate']=0.005
    mdp_demo_policy_dict = {}
    env_name = "LunarLander-v2"
    env_gym = gym.make(env_name)
    obs_size = len(env_gym.observation_space.high)
    env = GymMDP(env_name='LunarLander-v2', render=True, render_every_n_episodes=20)
    test_mdp = env #test mdp is the same
    mdp_demo_policy_dict[env]=lpd.expert_lunar_policy

    # ============================
    # == Make State Abstraction ==
    # ============================
    sess = tf.Session()
    nn_sa_file_name = "lunar_nn_sa"
    num_iterations = 300
    abstraction_net = make_nn_sa(mdp_demo_policy_dict, sess,params)
    nn_sa = NNStateAbstr(abstraction_net)

    # =================
    # == Make Agents ==
    # =================
    actions = test_mdp.get_actions()
    num_features = test_mdp.get_num_state_feats()
    linear_agent = LinearQAgent(actions=actions, num_features=num_features, alpha=params['rl_learning_rate'])
    sa_agent = AbstractionWrapper(QLearningAgent, agent_params={"alpha":params['rl_learning_rate'],"actions":test_mdp.get_actions(), "anneal":True}, state_abstr=nn_sa, name_ext="$-\\phi$")

    # ====================
    # == Run Experiment ==
    # ====================
    run_agents_on_mdp([sa_agent, linear_agent], test_mdp, instances=params['num_instances'], episodes=params['episodes'], steps=params['steps'], verbose=True, track_success=True, success_reward=100)

if __name__ == "__main__":
    main()
