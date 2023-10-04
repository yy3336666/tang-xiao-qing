def ProbabilityOnY(data, label, attr, value, memo = {}):
	key = (label, attr, value)
	if key in memo:
		return memo[key]
	Y = data.iloc[:,-1] == label
	X = data[attr][Y]
	mu = np.mean(X)
	sigma = np.std(X)
	p = prob(mu, sigma)(value) #prob根据mu, sigma生成概率密度函数
	memo[key] = p
	return p
