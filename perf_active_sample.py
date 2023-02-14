import numpy as np
import collections
import pandas
result_dict = collections.defaultdict(list)


# root = './logs/fair_sampling/vit/'
# root = './logs/fair_sampling/res18/'

# dataset = 'compas'
# tol = 0.05
# avg_cnt = 3

dataset = 'celeba'
tol = 0.05
avg_cnt = 3


root = f'./logs/fair_sampling/{dataset}/'


def extract_line(line):
    line_ = line.strip('\n').split('|')
    # try:
    acc = float(line_[-2].strip(' ').split(' ')[2])
    # except:
    #     pass
    fair = float(line_[-1].strip(' ').split(' ')[3])
    return acc, fair


def get_result(file_name):
    
    if 'vit' in root:
        remove = 1
    else:
        remove = 1
    with open(root+file_name+'.log') as file:
        test_list = []
        val_list = []
        warm_list_t, warm_list_v = [], []
        idx_test, idx_val, idx_warm_train, idx_warm_val = -remove, -remove, -remove, -remove
        for line in file.readlines():
            if '|test' in line:
                acc, fair = extract_line(line)
                test_list.append((acc, fair, idx_test))
                idx_test += 1
            elif '|val' in line:
                acc, fair = extract_line(line)
                val_list.append((acc, fair, idx_val))
                idx_val += 1
            elif '|warm_t' in line:
                acc, fair = extract_line(line)
                warm_list_t.append((acc, fair, idx_warm_train))
                idx_warm_train += 1
            elif '|warm_v' in line:
                acc, fair = extract_line(line)
                warm_list_v.append((acc, fair, idx_warm_val))
                idx_warm_val += 1


    base_perf_t = warm_list_t[-avg_cnt:]
    base_perf_v = warm_list_v[-avg_cnt:]
    avg_acc_base_v = np.mean([i[0] for i in base_perf_v])
    avg_acc_base_t = np.mean([i[0] for i in base_perf_t])
    avg_fair_base_t = np.mean([i[1] for i in base_perf_t])

    test_list = test_list[remove:]
    val_list = val_list[remove:]
    if len(test_list) == len(val_list):
        # find models whose val_acc > avg_acc_base_v
        val_list_filtered = []
        for model_i in val_list:
            if model_i[0] >= avg_acc_base_v - tol:
                val_list_filtered.append(model_i)
        
        if len(val_list_filtered) >= avg_cnt:
            val_list = val_list_filtered
            print(f'[{file_name}] find top-{avg_cnt} (out of {len(val_list)}) fair models whose val_acc > avg_acc_base_v')
            sel = 1 # fair-focused
        else:
            sel = 0 # acc-focused
            print(f'[{file_name}] find top-{avg_cnt} accurate models')


            # fair_val = sorted(val_list_filtered, key=lambda x: x[1])



        # select top k by acc and fair, respectively
        k = avg_cnt
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
        result_dict[file_name] = [f'({acc_avg_sel_by_acc_val:.3f}, {fair_avg_sel_by_acc_val:.3f})', f'({acc_avg_sel_by_fair_val:.3f}, {fair_avg_sel_by_fair_val:.3f})', sel] # [acc_focused, fair_focused, fair_and_better_than_base_acc]       
        return f'({avg_acc_base_t:.3f}, {avg_fair_base_t:.3f})'
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
    elif focus == 'fair':
        sel = 1
    else:
        sel = None
    
    result = []
    for stg in strategy:
        for layer in sel_layers:
            # if stg == 1 and layer == 4:
            #     break
            rec = []
            for label in label_key:
                for metric in metrics:
                    # if stg == 0:
                    #     file_name = f'{label}'
                    # else:
                    file_name = f'{label}_s{stg}_{metric}_{layer}'
                    if 'res18' in root:
                        file_name = 'res18_' + file_name
                    if stg == 0:
                        rec.append(result_dict[file_name][0])
                    else:
                        if sel is not None:
                            rec.append(result_dict[file_name][sel]) # acc_focused: 0, fairness focused: 1
                        else:
                            rec.append(result_dict[file_name][result_dict[file_name][2]])
            result.append(rec)
    # print(result)
    
    file_path = f'result_{dataset}_{focus}_focused.csv'

    print_result(result, file_path)


sel_layers = [4]
strategy = [0, 1, 2, 5]
if dataset == 'celeba':
    label_key = ['Smiling', 'Straight_Hair', 'Attractive', 'Pale_Skin', 'Young', 'Big_Nose']
elif dataset == 'compas':
    label_key = ['label']
else:
    raise NameError(f'undefined dataset {dataset}')
metrics = ['dp', 'eop', 'eod']

# read logs, then save the processed results to dict
for layer in sel_layers:
    for stg in strategy:
        for label in label_key:
            for metric in metrics:
                # if stg == 1:
                #     file_name = f'{label}_s{stg}_{metric}_2'
                # else:
                if stg > 0:
                    file_name = f'{label}_s{stg}_{metric}_{layer}'
                    if 'res18' in root:
                        file_name = 'res18_' + file_name
                    base_pref = get_result(file_name)
                    result_dict[f'{label}_s0_{metric}_{layer}'] = [base_pref]
        
# get table
# get_table(focus = 'acc')
# get_table(focus = 'fair')
get_table(focus = f'auto_tol{tol}')
# print(result_dict)

