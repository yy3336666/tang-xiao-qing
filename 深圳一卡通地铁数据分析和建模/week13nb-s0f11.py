from numpy import sqrt, pi, exp
def prob(mu, sigma):
    def calprob(val):
        return 1/sqrt(2 * pi * sigma**2)*exp(-(mu-val)**2/(2*sigma**2))
    return calprob
def ProbabilityOnY(data, label, attr, value):
	Y = data.iloc[:,-1] == label
	X = data[attr][Y]
	mu = np.mean(X)
	sigma = np.std(X)
	return prob(mu, sigma)(value) #prob根据mu, sigma生成概率密度函数
