import keras
import sys
from keras.models import model_from_json
import numpy


# load json and create model
json_file = open('../mac/learned_policy/LunarLander-v2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model

def load_weights():
	loaded_model.load_weights('../mac/learned_policy/LunarLander-v2.h5')


def expert_lunar_policy(state):
	s_size=len(state.data)
	s_array=numpy.array(state.data).reshape(1,s_size)
	temp=loaded_model.predict(s_array)
	#print(temp)
	return numpy.argmax(temp)