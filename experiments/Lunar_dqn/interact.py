import gym
import numpy
import tensorflow as tf
import sys
env=gym.make("LunarLander-v2")



with tf.Session() as sess:
	imported_meta = tf.train.import_meta_graph('models/my_dqn_model-1000.meta')
	graph = tf.get_default_graph()
	imported_meta.restore(sess, 'models/my_dqn_model-1000')
	graph = tf.get_default_graph()
	for op in graph.get_operations():
		print(op.name)
	state=graph.get_tensor_by_name("Placeholder:0")
	is_training_ph=graph.get_tensor_by_name("Placeholder_5:0")
	y_pred = graph.get_tensor_by_name("q_network/dense_3/BiasAdd:0")
	#sys.exit(1)
	G_li=[]
	for ep in range(100):
		s=env.reset()
		done=False
		G=0
		while True:
			y=sess.run(y_pred,feed_dict={ state:numpy.array(s).reshape(1,8),is_training_ph:False })
			q_val=y[0]
			a_max=numpy.argmax(q_val)
			sp,r,done,_=env.step(a_max)
			G=G+r
			env.render()
			s=sp
			if done==True:
				G_li.append(G)
				#print(G)
				#print(numpy.mean(G_li))
				break
