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

#creates bandit where the mean of each arm changes according to func
class ChangingBandit:
	def __init__(self, func, numArms= 10, mean = 0, std = 1, armStd = 1):
		self.arms = np.random.normal(mean, std, (numArms))
		self.pulls = np.zeros(numArms)
		self.armStd = armStd
		self.func = func
	#arm -> index of which arm has been chosen to pull
	def PullArm(self, arm):
		self.pulls[arm] += 1
		return np.random.normal(self.arms[arm] + self.func(self.pulls[arm]), self.armStd)


class EpsGreedy:
	#eps -> probability the algorithm chooses to explore a random arm rather than choosing the currently known best arm
	def __init__(self, numArms = 10, mean = 0, std = 1, armStd = 1, eps = .1, learnRate = -1, func = None):
		if func == None:
			self.bandit = Bandit(numArms, mean, std, armStd) #the bandit
		else:
			self.bandit = ChangingBandit(func, numArms, mean, std, armStd)
		self.averages = np.zeros(numArms) #array containing currently approximated expected reward from each arm
		self.actionCount = np.zeros(numArms) #keeps track of how many times each arm has been chosen
		self.currentStep = 0 #current iteration
		self.totalReward = 0 #cumulate model reward
		self.averageReward = None #array containing average model reward as the model learns
		self.averagsOverTime = None #array containing the approximated expected reward from each arm over time
		self.eps = eps #probability the model chooses a random arm rather than the best known arm
		self.learnRate = learnRate

	def AdjustExpectedReward(self, arm, reward):
		if self.learnRate <= 0:
			self.averages[arm] = self.averages[arm] + (1/self.actionCount[arm])*(reward - self.averages[arm])
		else:
			self.averages[arm] = self.averages[arm] + self.learnRate*(reward - self.averages[arm])

	def Learn(self, numIterations = 1000):
		self.averageReward = np.zeros(numIterations)
		self.averagesOverTime = np.zeros(shape = (numIterations, len(self.averages)))
		while self.currentStep < numIterations:
			if np.random.random() < self.eps:
				action = np.random.randint(0,len(self.averages))
			else:
				action = np.argmax(self.averages)
			reward = self.bandit.PullArm(action)

			self.totalReward += reward
			self.currentStep += 1
			self.actionCount[action] += 1

			self.averageReward[self.currentStep-1] = self.totalReward / self.currentStep
			self.AdjustExpectedReward(action, reward)
			self.averagesOverTime[self.currentStep-1] = self.averages

	def PlotProgress(self):
		maxReward = np.zeros(len(self.averageReward)) + np.max(self.bandit.arms)
		plt.plot(self.averageReward, label = "Average: " + str(self.averageReward[-1]))
		plt.plot(maxReward, label = "Average Reward From Optimal Arm")
		plt.xlabel("Iterations")
		plt.ylabel("Average Reward")
		plt.title("Learning Progress of EpsGreedy Algorithm with Eps = " + str(self.eps))
		plt.legend()
		plt.show()

	def PlotExpectedReward(self):
		temp = np.transpose(self.averagesOverTime)
		colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
			'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
		for i in range(len(self.averages)):
			plt.plot(temp[i], label = "Estimated " + str(i), color = colors[i % len(colors)])
			plt.plot(self.bandit.arms[i] + np.zeros(len(temp[i])), label = "Actual " + str(i), color = colors[i % len(colors)])
		plt.xlabel("Iterations")
		plt.ylabel("Average Reward")
		plt.title("Model Attempting to Approximate Arm Rewards, Learn Rate: " + str(self.learnRate))
		plt.legend()
		plt.show()

test = EpsGreedy(learnRate = .1, func = lambda x: x/100)
test.Learn()
#test.PlotProgress()
test.PlotExpectedReward()