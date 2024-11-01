import numpy as np
import collections
import pandas
result_dict = collections.defaultdict(list)

result_dict_val = collections.defaultdict(list)



'''
Dataset option
'''

# 1: adult
# 2: compas
# 3: jigsaw
# 4: celeba

dataset_type = 'adult'
runs=0  #random seed


if dataset_type == 'adult':

    tol = 0.05
    avg_cnt = 3 
    suffix = ''
    senstive_attrbutes = 'age' #sex
    root = f'./logs/fair_sampling/{dataset_type}-{senstive_attrbutes}-runs{runs}/'
    
elif dataset_type == 'compas':
    tol = 0.05
    avg_cnt = 3
    suffix = ''
    root = f'./logs/fair_sampling/{dataset_type}-runs{runs}/'

elif dataset_type == 'jigsaw':
    tol = 0.05
    avg_cnt = 3
    suffix = ''
    root = f'./logs/fair_sampling/{dataset_type}-runs{runs}/'

elif dataset_type == 'celeba':
    tol = 0.05
    avg_cnt = 3
    suffix = ''
    root = f'./logs/fair_sampling/{dataset_type}-runs{runs}/'



def extract_line(line):
    line_ = line.strip('\n').split('|')
    acc = float(line_[-2].strip(' ').split(' ')[2])
    fair = float(line_[-1].strip(' ').split(' ')[3])
    return acc, fair


def get_result_val(file_name, val_length):
    
    if 'vit' in root:
        remove = 1
    else:
        remove = 1
    with open(root+file_name + '.log') as file:
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

    test_list = test_list[remove:val_length]
    val_list = val_list[remove:val_length]
    if len(test_list) == len(val_list):
        
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

        # select top k by acc and fair, respectively
        k = 3
        acc_val = sorted(val_list, key=lambda x: x[0])[::-1]
        fair_val = sorted(val_list, key=lambda x: x[1])

        # get average
        acc_avg_sel_by_fair_val = np.mean([test_list[i[2]][0] for i in fair_val[:k]])
        fair_avg_sel_by_fair_val = np.mean([test_list[i[2]][1] for i in fair_val[:k]])


        acc_avg_sel_by_acc_val = np.mean([test_list[i[2]][0] for i in acc_val[:k]])
        fair_avg_sel_by_acc_val = np.mean([test_list[i[2]][1] for i in acc_val[:k]])

        # save result as a dict
        result_dict[file_name] = [f'({acc_avg_sel_by_acc_val:.3f}, {fair_avg_sel_by_acc_val:.3f})', f'({acc_avg_sel_by_fair_val:.3f}, {fair_avg_sel_by_fair_val:.3f})', sel] # [acc_focused, fair_focused, fair_and_better_than_base_acc]       
        # Round to 3 decimal places
        acc_rounded = round(acc_avg_sel_by_fair_val, 3)
        fair_rounded = round(fair_avg_sel_by_fair_val, 3)

        # Create a tuple
        result_tuple = (acc_rounded, fair_rounded)
        return result_tuple

    else:
        raise RuntimeError('test_list has a different length from val_list')
    



def get_result_typical(file_name):
    
    if 'vit' in root:
        remove = 1
    else:
        remove = 1
    with open(root+file_name + '.log') as file:
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

        # select top k by acc and fair, respectively
        k = avg_cnt
        acc_val = sorted(val_list, key=lambda x: x[0])[::-1]
        fair_val = sorted(val_list, key=lambda x: x[1])
        
        # get average
        acc_avg_sel_by_fair_val = np.mean([test_list[i[2]][0] for i in fair_val[:k]])

        fair_avg_sel_by_fair_val = np.mean([test_list[i[2]][1] for i in fair_val[:k]])
        acc_avg_sel_by_acc_val = np.mean([test_list[i[2]][0] for i in acc_val[:k]])
        fair_avg_sel_by_acc_val = np.mean([test_list[i[2]][1] for i in acc_val[:k]])

        # save result as a dict
        result_dict[file_name] = [f'({acc_avg_sel_by_acc_val:.3f}, {fair_avg_sel_by_acc_val:.3f})', f'({acc_avg_sel_by_fair_val:.3f}, {fair_avg_sel_by_fair_val:.3f})', sel] # [acc_focused, fair_focused, fair_and_better_than_base_acc]       
        return f'({avg_acc_base_t:.3f}, {avg_fair_base_t:.3f})'
    else:
        raise RuntimeError('test_list has a different length from val_list')

def print_result(result, file_path):
    df = pandas.DataFrame(result)
    print('this is the information of table!!')
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
            rec = []
            for label in label_key:
                for metric in metrics:
                    for val_ratio in val_ratios:

                        file_name = f'{label}_s{stg}_{metric}_{layer}' + '_'+ label_tag + '_' + f'{val_ratio}'
                        if 'res18' in root:
                            file_name = 'res18_' + file_name
                        if suffix:
                            file_name += suffix
                        if stg == 0:
                            rec.append(result_dict[file_name][0])
                        else:
                            if sel is not None:
                                rec.append(result_dict[file_name][sel]) # acc_focused: 0, fairness focused: 1
                            else:
                                rec.append(result_dict[file_name][result_dict[file_name][2]])
            result.append(rec)
    
    file_path = f'result_{dataset_type}_{focus}_focused_{suffix}.csv'

    print_result(result, file_path)




def get_val_length(file_name):
    
    if 'vit' in root:
        remove = 1
    else:
        remove = 1
    with open(root+file_name + '.log') as file:
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

    return idx_val




###########################################################################################
######## table results setting
###########################################################################################

sel_layers = [4] # selected layers
# select BALD as the baseline to compare
strategy = [0, 1, 6, 8, 7, 2, 5]
metrics = ['dp', 'eop', 'eod']

if dataset_type == 'celeba':
    label_key = [ 'Young', 'Big_Nose''Smiling', 'Attractive']
    val_ratios = [0.02, 0.05, 0.1] 
elif dataset_type == 'compas':
    label_key = ['label']
    val_ratios = [0.01, 0.05, 0.1, 0.2, 0.25, 0.5] 

elif dataset_type == 'adult':
    label_key = ['label']
    val_ratios = [0.01, 0.05, 0.1, 0.2, 0.25, 0.5]  

elif dataset_type == 'jigsaw':
    label_key = ['label']
else:
    raise NameError(f'undefined dataset {dataset_type}')

# take one fairness definition to compare 

label_tag = 'default'
# read logs, then save the processed results to dict
for layer in sel_layers:
    for stg in strategy:
        for label in label_key:
            for metric in metrics:
                for val_ratio in val_ratios:
        
                    if stg > 0:
                        file_name = f'{label}_s{stg}_{metric}_{layer}' + '_'+ label_tag + '_' + f'{val_ratio}'
                        if 'res18' in root:
                            file_name = 'res18_' + file_name
                        if suffix:
                            file_name += suffix
                        base_pref = get_result_typical(file_name)
                        result_dict[f'{label}_s0_{metric}_{layer}' + '_'+ label_tag + suffix] = [base_pref]
            
get_table(focus = f'auto_tol{tol}')
