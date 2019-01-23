from gym.envs.registration import registry, register, make, spec
register(
    id='LunarNoShaping-v0',
    entry_point='lunar_variants.lunar_no_shaping:LunarLanderNew',
)
'''
kwargs={
    'model_name': 'cartpole_learned_k'+str(k)
}
'''