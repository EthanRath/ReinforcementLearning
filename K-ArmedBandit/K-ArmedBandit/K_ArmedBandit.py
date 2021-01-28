from matplotlib import pyplot as plt
import numpy as np
import random

class Bandit:
	#numArms -> number of arms for the bandit
	#mean,std -> parameters for the gaussian distribution that determines the average reward given by each arm
	#armStd -> standard deviation of reward when a given arm is chosen
	def __init__(self, numArms = 10, mean = 0, std = 1, armStd = 1):
		self.arms = np.random.normal(mean, std, (numArms))
		self.armStd = armStd
	#arm -> index of which arm has been chosen to pull
	def PullArm(self, arm):
		return np.random.normal(self.arms[arm], self.armStd)

class EpsGreedy:
	def __init__(self, numArms = 10, mean = 0, std = 1, armStd = 1, eps = .1):
		self.bandit = Bandit(numArms, mean, std, armStd)
		self.averages = np.zeros(numArms)
		self.currentStep = 0
		self.totalReward = 0
		self.averageReward = None
		self.eps = eps

	def AdjustExpectedReward(self, arm, reward):
		self.averages[arm] = self.averages[arm] + (1/self.currentStep)*(reward - self.averages[arm])

	def Learn(self, numIterations = 1000):
		self.averageReward = np.zeros(numIterations)
		while self.currentStep < numIterations:
			if np.random.random() < self.eps:
				action = np.random.randint(0,len(self.averages))
			else:
				action = np.argmax(self.averages)
			reward = self.bandit.PullArm(action)
			self.totalReward += reward
			self.currentStep += 1
			self.averageReward[self.currentStep-1] = self.totalReward / self.currentStep
			self.AdjustExpectedReward(action, reward)

	def PlotProgress(self):
		plt.plot(self.averageReward, label = "Average: " + str(self.averageReward[-1]))
		plt.xlabel("Iterations")
		plt.ylabel("Average Reward")
		plt.title("Learning Progress of EpsGreedy Algorithm with Eps = " + str(self.eps))
		plt.legend()
		plt.show()

test = EpsGreedy()
test.Learn()
test.PlotProgress()