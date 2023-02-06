import numpy as np
import collections
import pandas
result_dict = collections.defaultdict(list)


root = './logs/fair_sampling/vit/'


def get_result(file_name):
    
    with open(root+file_name+'.log') as file:
        test_list = []
        val_list = []
        idx_test, idx_val = -1, -1
        for line in file.readlines():
            if 'test' in line:
                line_ = line.strip('\n').split('|')
                acc = float(line_[-2].strip(' ').split(' ')[2])
                fair = float(line_[-1].strip(' ').split(' ')[3])
                test_list.append((acc, fair, idx_test))
                idx_test += 1
            elif 'val' in line:
                line_ = line.strip('\n').split('|')
                acc = float(line_[-2].strip(' ').split(' ')[2])
                fair = float(line_[-1].strip(' ').split(' ')[3])
                val_list.append((acc, fair, idx_val))
                idx_val += 1
    test_list = test_list[1:]
    val_list = val_list[1:]
    if len(test_list) == len(val_list):
        # select top k by acc and fair, respectively
        k = 2
        acc_val = sorted(val_list, key=lambda x: x[0])[::-1]
        fair_val = sorted(val_list, key=lambda x: x[1])
        # print(acc_val)
        # print(fair_val)
        # get average
        acc_avg_sel_by_fair_val = np.mean([test_list[i[2]][0] for i in fair_val[:k]])
        fair_avg_sel_by_fair_val = np.mean([test_list[i[2]][1] for i in fair_val[:k]])
        acc_avg_sel_by_acc_val = np.mean([test_list[i[2]][0] for i in acc_val[:k]])
        fair_avg_sel_by_acc_val = np.mean([test_list[i[2]][1] for i in acc_val[:k]])

        # save result as a dict
        # result_dict[file_name] = [(acc_avg_sel_by_acc_val, fair_avg_sel_by_acc_val), (acc_avg_sel_by_fair_val, fair_avg_sel_by_fair_val)] # [acc_focused, fair_focused]     
        result_dict[file_name] = [f'({acc_avg_sel_by_acc_val:.3f}, {fair_avg_sel_by_acc_val:.3f})', f'({acc_avg_sel_by_fair_val:.3f}, {fair_avg_sel_by_fair_val:.3f})'] # [acc_focused, fair_focused]          
    else:
        raise RuntimeError('test_list has a different length from val_list')

def print_result(result, file_path):
    df = pandas.DataFrame(result)
    print(df)
    df.to_csv(file_path, index=False)
    print(f'result is saved to {file_path}')


def get_table(focus):
    if focus == 'acc':
        sel = 0
    else:
        sel = 1
    result = []
    for stg in strategy:
        for layer in sel_layers:
            if stg == 1 and layer == 4:
                break
            rec = []
            for label in label_key:
                for metric in metrics:
                    file_name = f'{label}_s{stg}_{metric}_{layer}'
                    rec.append(result_dict[file_name][sel]) # acc_focused: 0, fairness focused: 1
            result.append(rec)
    # print(result)
    file_path = f'result_{focus}_focused.csv'
    print_result(result, file_path)

sel_layers = [2, 4]
strategy = [1, 2, 5]
label_key = ['Smiling', 'Straight_Hair', 'Attractive']
metrics = ['dp', 'eop', 'eod']

# read logs, then save the processed results to dict
for layer in sel_layers:
    for stg in strategy:
        for label in label_key:
            for metric in metrics:
                if stg == 1:
                    file_name = f'{label}_s{stg}_{metric}_2'
                else:
                    file_name = f'{label}_s{stg}_{metric}_{layer}'
                get_result(file_name)
        
# get table
get_table(focus = 'acc')
get_table(focus = 'fairness')
# print(result_dict)

