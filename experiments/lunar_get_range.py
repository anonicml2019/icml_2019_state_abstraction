import numpy, gym, random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

env=gym.make("LunarLander-v2")
episode_max=100
s_li=[]

for e in range(episode_max):
	env.reset()
	Done=False
	while Done==False:
		a=random.randint(0,3)
		s,_,Done,_=env.step(a)
		s_li.append(s)
		#print(s)



for index in range(8):
	plt.subplot(str('33')+str(index+1))
	x_li=[s[index] for s in s_li]
	sns.dist(x_li)
plt.show()