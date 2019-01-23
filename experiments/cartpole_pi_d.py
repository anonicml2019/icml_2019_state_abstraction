import keras
import sys
from keras.models import model_from_json
import numpy


# load json and create model
json_file = open('../mac/learned_policy/CartPole-v0.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model

# def load_weights():
loaded_model.load_weights('../mac/learned_policy/CartPole-v0.h5')

def expert_cartpole_policy(state):
	s_size=len(state)
	s_array=numpy.array(state).reshape(1,s_size)
	temp=loaded_model.predict(s_array)
	# return numpy.random.choice(a=[0,1], p=temp[0])
	#print(temp[0])
	return numpy.argmax(temp[0])