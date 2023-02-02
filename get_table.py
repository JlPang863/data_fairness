import numpy as np
import collections
result_dict = collections.defaultdict(list)
cur_name = ''


def print_result(acc, dp):
    for i in range(len(acc)):
        print(acc[i] + dp[i])



with open('result.log') as file:
    for line in file.readlines():
        line_ = line.strip('\n').split(' ')
        if len(line_) == 1:
            cur_name = line_[0]
        else:
            result_dict[cur_name] = [float(i) for i in line_]

# print(result_dict)

exps = [f'exp{i+1}' for i in range(3)]
cols = ['no_conf', 'entropy', 'peer']
tvs = ['TV', 'V']


val = 0.2
train = 0.02

acc = []
dp = []

for val in [0.2, 0.4, 0.8]:

    for exp in exps:
        acc.append([])
        dp.append([])
        for tv in tvs:
            for col in cols:
                if col == 'no_conf':
                    name = f'{val}_{train}_{col}_{exp}_TV'
                else:
                    name = f'{val}_{train}_{col}_{exp}_{tv}'
                if tv == 'V' and col == 'no_conf':
                    pass
                else:
                    # print(name)
                    if len(result_dict[name]) > 0:
                        acc[-1].append(result_dict[name][0])
                        dp[-1].append(result_dict[name][1])
                    else:
                        acc[-1].append('-')
                        dp[-1].append('-')




print_result(acc, dp)



# print(acc)
# print(dp)


    
