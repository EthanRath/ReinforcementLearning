import numpy as np
import math
import random
from matplotlib import pyplot as plt

class Game:
	def __init__(self, dims, initializer):
		#key_positions[0] is a list of np arrays containing indices of locations that send you to another location
		#key_positions[1] holds the corresponding destination of said locations
		self.board, self.position, self.key_positions = initializer(dims)

	def Move(self, direction):
		old_pos = self.position
		idx = self.Check_Key()
		if idx >= 0:
			self.position = self.key_positions[1][idx]
			#return self.key_positions[2][idx]
		else:
			new_pos = self.position + direction
			dims = np.array(self.board.shape)-1
			if (new_pos > dims).any() or (new_pos < 0).any():
				return -1
			self.position = new_pos
		return self.board[tuple(old_pos)]

	def Check_Key(self):
		for i in range(len(self.key_positions[0])):
			key = self.key_positions[0][i]
			if (key == self.position).all():
				return i
		return -1
		
class TD:
	def __init__(self, game, learn_rate, discount_rate, policy):
		self.game = game
		self.V = np.random.uniform(size = game.board.shape)
		self.lr = learn_rate
		self.dr = discount_rate
		self.policy = policy

	def Move(self):
		#directions = [[1,0], [0,1], [-1,0], [0,-1]]
		#direction = directions[random.randint(0,3)]
		direction = self.policy(self.V, self.game)
		old_pos = self.game.position

		reward = self.game.Move(direction)
		new_state_score = self.V[tuple(self.game.position)]

		self.V[tuple(old_pos)] += self.lr * (reward + (self.dr*new_state_score) - self.V[tuple(old_pos)])

	def Learn(self, steps, actual = []):
		mse = []
		for i in range(steps):
			if len(actual) > 0:
				mse += [self.MSE(actual)]
			self.Move()
		if len(actual) > 0:
				mse += [self.MSE(actual)]
		return mse

	def MSE(self, actual):
		return np.mean((self.V - actual)**2)

	def Plot(self, mse):
		plt.plot(np.arange(0, len(mse), 1), mse)
		plt.show()

def Optimize_Policy(V, game):
	directions = [[1,0], [0,1], [-1,0], [0,-1]]
	pos = game.position
	dims = np.array(game.board.shape)-1
	score = [0,0,0,0]
	for i in range(len(directions)):
		new_pos = pos + np.array(directions[i])
		if (new_pos > dims).any() or (new_pos < 0).any():
				score[i] = -1
		else:
			score[i] = V[tuple(new_pos)]
	score = np.array(score)
	total = np.sum((score>0)*score)
	score = score / total
	r = random.random()
	sum = 0
	for i in range(len(score)):
		if score[i] < 0:
			continue
		temp = score[i]
		score[i] += sum
		if score[i] >= r:
			return directions[i]
		sum += temp
	print("Something went wrong")
	print(r)
	print(score)
	print(total)
	print(sum)
		
def Optimal_Policy(V, game, pos):
	directions = [[1,0], [0,1], [-1,0], [0,-1]]
	dims = np.array(game.board.shape)-1
	score = [0,0,0,0]
	for i in range(len(directions)):
		new_pos = pos + np.array(directions[i])
		if (new_pos > dims).any() or (new_pos < 0).any():
				score[i] = -1
		else:
			score[i] = V[tuple(new_pos)]
	score = np.array(score)
	total = np.sum((score>0)*score)
	score = score / total
	print(score)
	return np.argmax(score)

def Random_Policy(V, game):
	directions = [[1,0], [0,1], [-1,0], [0,-1]]
	direction = directions[random.randint(0,3)]
	return direction

def Init_Book(dims):
	board = np.zeros(shape = (5,5))

	board[0,1] = 10
	board[0,3] = 5

	key_positions = [[np.array([0,1]), np.array([0,3])], [np.array([4,1]), np.array([2, 3])]]

	iniz = np.array([2,2])

	return board, iniz, key_positions

def Test_Random():
	game = Game([0,0], Init_Book)
	T = TD(game, .05, .75, Random_Policy)
	actual = np.array([[3.3,8.8,4.4,5.3,1.5], [1.5,3.0,2.3,1.9,.5], [.1,.7,.7,.4,-.4], [-1,-.4,-.4,-.6,-1.2], [-1.9,-1.3,-1.2,-1.4,-2]])
	mse = T.Learn(100000,actual)
	print(T.V)
	T.Plot(mse)

def Test_Optimal():
	game = Game([0,0], Init_Book)
	T = TD(game, .05, .75, Optimize_Policy)
	actual = np.array([[22,24,22,19.4,17.5], [19.8,22,19.8,17.8,16], [17.8,19.8,17.8,16,14.4], [16,17.8,16,14.4,13], [14.4,16,14.4,13,11.7]])
	mse = T.Learn(100000,actual)
	print(T.V)

	#It will be hard to read the graph but it goes like this:
	#0 = down, 1 = right, 2 = up, 3 = left
	#for the points (0,1) and (0,3) the direction chosen doesn't matter because the agent is automatically moved to a predetermined position
	policy = np.zeros(shape = (5,5))
	for i in range(5):
		for j in range(5):
			policy[i,j] = Optimal_Policy(T.V, T.game, np.array([i,j]))
	print(policy)
	plt.imshow(policy)
	plt.show()

	T.Plot(mse)

if __name__ == "__main__":
	#Test_Random()
	Test_Optimal()
	

