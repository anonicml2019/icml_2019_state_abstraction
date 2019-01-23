import tensorflow as tf
import gym
import numpy


class l_p():
	def __init__(self):
		self.graph=tf.Graph()
		self.sess = tf.Session(graph=self.graph)
		with self.graph.as_default():
			saver = tf.train.import_meta_graph('Lunar_dqn/models/my_dqn_model-1000.meta',clear_devices=True)
			saver.restore(self.sess, 'Lunar_dqn/models/my_dqn_model-1000')
			self.state=self.graph.get_tensor_by_name("Placeholder:0")
			self.is_training_ph=self.graph.get_tensor_by_name("Placeholder_5:0")
			self.Q = self.graph.get_tensor_by_name("q_network/dense_3/BiasAdd:0")

	def get_q_values(self,state):
		state=state.reshape(1,len(state))
		return self.sess.run(self.Q,feed_dict={self.state:state,self.is_training_ph:False })

	def get_a_star(self,state):
		vals=self.get_q_values(state)[0]
		return numpy.argmax(vals)

'''
env=gym.make("LunarLander-v2")
s=env.reset()
policy=l_p()
a=policy.get_a_star(s)
'''