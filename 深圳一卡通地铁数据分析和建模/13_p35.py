def ProbabilityOnY(data , label , attr , value , lamb = 1, memo = {}):
    key = (label, attr, value)
    if key in memo:
        return memo[key]
    Y = data.iloc[:,-1] == label
    A = data[attr] == value
    S = len(set(data[attr]))
    p = (sum(Y & A) + lamb) / (sum(Y) + lamb * S)
    memo[key] = p
    return p

def ProbabilityJoin(data , label , lamb = 1, memo = {} , ** argv):
    if label in memo:
        P = memo[label]
    else:
        Y = data.iloc[:,-1] #约定最后一列作为列标签
        K = len(set(Y))
        P = (sum(Y == label) + lamb)/ (len(Y) + K * lamb)
        memo[label] = P
    for attr in argv.keys():
        p = ProbabilityOnY(data , label , attr , argv[attr])
        P *= p
    return P