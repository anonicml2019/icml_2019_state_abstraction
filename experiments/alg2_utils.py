import numpy
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_learned_abstraction(abstraction_net, fig, ax, num_clusters):
	samples=[]
	for x in numpy.arange(0,5,.25):
		for y in numpy.arange(0,5,.25):
			samples.append(([x,y],0,0))

	p_li=abstraction_net.predict(samples)
	print(samples[-1])
	print(p_li[-1])
	c_li=[numpy.argmax(p) for p in p_li]
	s_li=[200*numpy.abs(numpy.max(p)-1./num_clusters) for p in p_li]
	colors=['green','red','blue','black']
	li_x_coordinates=[[] for _ in range(num_clusters)]
	li_y_coordinates=[[] for _ in range(num_clusters)]
	li_size=[[] for _ in range(num_clusters)]
	for index, x in enumerate(samples):
		li_x_coordinates[c_li[index]].append(x[0][0])
		li_y_coordinates[c_li[index]].append(x[0][1])
		li_size[c_li[index]].append(s_li[index])

	for c_number in range(num_clusters):
		ax.scatter(li_x_coordinates[c_number],li_y_coordinates[c_number],
				   color=colors[c_number],s=li_size[c_number])
	

	x_li=[x/100. for x in range(500)]
	y_li=[x*x for x in x_li]
	y2_li=[numpy.sqrt(x)-1 for x in x_li]
	y3_li=[numpy.sqrt(30-x*x) for x in x_li]
	plt.plot(x_li,y_li,color='black')
	plt.plot(x_li,y2_li,color='black')
	plt.plot(x_li,y3_li,color='black')

	plt.ylim([0,5])
	plt.xlim([0,4.75])
	fig.canvas.draw_idle()
	plt.pause(0.1)
	plt.cla() 


def demonstrator(num_samples,num_mdps):
	samples=[]
	for mdp_index in range(num_mdps):
		for _ in range(num_samples):
			s=5*numpy.random.random(2)# demonstrator needs to properly implement this ...
			
			if s[0]*s[0]<s[1]:
				a=0
			elif numpy.sqrt(s[0])-1>s[1]:
				if mdp_index==0:
					a=0
				elif mdp_index==1:
					a=2
			elif s[0]*s[0]+s[1]*s[1]>=30:
				a=3
			else:
				a=1
			samples.append((s,a,mdp_index))

	return samples

def grid_demo_policy(state, mdp_index=0):
	'''
	Args:
		state

	Returns:
		(str)
	'''

	s = 5 * numpy.random.random(2)# demonstrator needs to properly implement this ...
	
	if s[0] * s[0] < s[1]:
		a = 0
	elif numpy.sqrt(s[0]) -1 > s[1]:
		if mdp_index == 0:
			a = 0
		elif mdp_index == 1:
			a = 2
	elif s[0] * s[0] + s[1] * s[1] >= 30:
		a = 3
	else:
		a = 1

	return a


def collect_samples_from_demo_policy(mdp, demo_policy, num_samples, num_mdps=1):
	'''
	Args:
		mdp (simple_rl.MDP)
		demo_policy (lambda : simple_rl.State --> str)
		num_samples (int)

	Returns:
		(list): A collection of (s, a, mdp_id) tuples.
	'''
	samples = []
	for mdp_index in range(num_mdps):
		cur_state = mdp.get_init_state()
		for _ in range(num_samples):

			# Grab next sample.
			cur_a = demo_policy(cur_state)
			action_index = mdp.get_actions().index(cur_a)	
			samples.append((cur_state, action_index, mdp_index))

			# Transition.
			if cur_state.is_terminal():
				mdp.reset()
			else:
				cur_state = mdp.get_transition_func()(cur_state, cur_a)


		mdp.reset()

	return samples