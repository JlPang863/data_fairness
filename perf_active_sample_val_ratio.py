import numpy as np
import collections
import pandas
result_dict = collections.defaultdict(list)

result_dict_val = collections.defaultdict(list)

# root = './logs/fair_sampling/vit/'
# root = './logs/fair_sampling/res18/'




'''
Dataset option
'''

# 1: adult
# 2: compas
# 3: jigsaw
# 4: celeba

dataset_num = 1

### random seed ###
runs=2


if dataset_num == 1:
    ################ adult dataset ###########################

    dataset = 'adult'
    tol = 0.05
    avg_cnt = 30
    suffix = ''
    # root = f'./logs/fair_sampling/{dataset}-age-runs2/'
    root = f'./logs/fair_sampling/{dataset}-sex-runs{runs}/'

    ###########################################################
elif dataset_num ==2:

    ################ compas dataset ###########################
    dataset = 'compas'
    tol = 0.05
    avg_cnt = 3
    suffix = ''
    root = f'./logs/fair_sampling/{dataset}-runs{runs}/'
    ###########################################################

elif dataset_num == 3:
    ################ jigsaw dataset ###########################

    dataset = 'jigsaw'
    tol = 0.05
    avg_cnt = 3
    suffix = ''
    root = f'./logs/fair_sampling/{dataset}-runs{runs}/'
    ###########################################################

elif dataset_num == 4:

    ############### celeba dataset ###########################
    dataset = 'celeba'
    tol = 0.05
    #tol = 0.01
    avg_cnt = 3
    suffix = ''
    # root = f'./logs/fair_sampling/{dataset}-backup/'
    # root = f'./logs/fair_sampling/{dataset}/'
    root = f'./logs/fair_sampling/{dataset}-runs{runs}/'
    # root = f'./logs/fair_sampling/{dataset}-01-27-runs0/'
    # root = f'.logs/fair_sampling/celeba-label budget-1024-ratio0.3/'




# suffix = '_prob_0.9_warm2batch_imgaux'


# suffix = '_tol0.02_wolb'
# suffix = '_prob_0.9'
# suffix = '_prob_0.95_warm2batch'
# suffix = '_prob_0.95_warm2batch_half_ablation'

# suffix = '_prob_0.99_warm2batch_scut_org_0.9_thre3'
###########################################################



def extract_line(line):
    line_ = line.strip('\n').split('|')
    # try:
    acc = float(line_[-2].strip(' ').split(' ')[2])
    # except:
    #     pass
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

    # print('##############get_result_val--- idx_val: ' +str(idx_val))

    base_perf_t = warm_list_t[-avg_cnt:]
    base_perf_v = warm_list_v[-avg_cnt:]
    # import pdb
    # pdb.set_trace()
    avg_acc_base_v = np.mean([i[0] for i in base_perf_v])
    avg_acc_base_t = np.mean([i[0] for i in base_perf_t])
    avg_fair_base_t = np.mean([i[1] for i in base_perf_t])

    # import pdb
    # pdb.set_trace()
    # print('##################test_list: ' + str(test_list))
    # print('##################avg_acc_base_v: ' + str(avg_acc_base_v - tol))
    test_list = test_list[remove:val_length]
    val_list = val_list[remove:val_length]
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

        # print('##################val_list_filtered: ' + str(val_list_filtered))

        # select top k by acc and fair, respectively
        k = 3
        acc_val = sorted(val_list, key=lambda x: x[0])[::-1]
        fair_val = sorted(val_list, key=lambda x: x[1])
        
        # k = val_length
        # acc_val = val_list
        # fair_val = val_list


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
        # return f'({acc_avg_sel_by_fair_val:.3f}, {fair_avg_sel_by_fair_val:.3f}), {val_length}'
        # return f'({acc_avg_sel_by_fair_val:.3f}, {fair_avg_sel_by_fair_val:.3f})'
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
    # import pdb
    # pdb.set_trace()
    avg_acc_base_v = np.mean([i[0] for i in base_perf_v])
    avg_acc_base_t = np.mean([i[0] for i in base_perf_t])
    avg_fair_base_t = np.mean([i[1] for i in base_perf_t])

    test_list = test_list[remove:]
    val_list = val_list[remove:]
    # import pdb
    # pdb.set_trace()
    # print('##################test_list: ' + str(test_list))
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

        # import pdb
        # pdb.set_trace()
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
            # if stg == 1 and layer == 4:
            #     break
            rec = []
            for label in label_key:
                for metric in metrics:
                    for val_ratio in val_ratios:

                        # if stg == 0:
                        #     file_name = f'{label}'
                        # else:
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
    # print(result)
    
    file_path = f'result_{dataset}_{focus}_focused_{suffix}.csv'

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


##########################################################################



###########################################################################################
'''
table results
'''
sel_layers = [4]
# default strategy: [0, 1, 6, 8, 7, 2, 5]
# select BALD as the baseline to compare
strategy = [2]#,
metrics = ['dp', 'eop', 'eod']

if dataset == 'celeba':
    # label_key = ['Smiling', 'Straight_Hair', 'Attractive', 'Pale_Skin', 'Young', 'Big_Nose']
    label_key = ['Smiling']#  'Young', 'Big_Nose''Smiling', 'Attractive',
    # label_key = ['Young']
    # label_key = ['Attractive']
    val_ratios = [0.02, 0.05, 0.1, 0.15]
    metrics = ['dp']
elif dataset == 'compas':
    label_key = ['label']
    val_ratios = [ 0.2, 0.25] #0.01, 0.05, 0.1, 0.2, 0.25, 0.5

elif dataset == 'adult':
    label_key = ['label']
    val_ratios = [0.1, 0.2, 0.25] # 0.01, 0.05, 0.1, 0.2, 0.25, 0.5

elif dataset == 'jigsaw':
    label_key = ['label']
else:
    raise NameError(f'undefined dataset {dataset}')
# take one fairness definition to compare 

label_tag = 'default'

# read logs, then save the processed results to dict
for layer in sel_layers:
    for stg in strategy:
        for label in label_key:
            for metric in metrics:
                for val_ratio in val_ratios:
                    # if stg == 1:
                    #     file_name = f'{label}_s{stg}_{metric}_2'
                    # else:
                    if stg > 0:
                        file_name = f'{label}_s{stg}_{metric}_{layer}' + '_'+ label_tag + '_' + f'{val_ratio}'
                        if 'res18' in root:
                            file_name = 'res18_' + file_name
                        if suffix:
                            file_name += suffix
                        base_pref = get_result_typical(file_name)
                        result_dict[f'{label}_s0_{metric}_{layer}' + '_'+ label_tag + suffix] = [base_pref]
            
# get table
# get_table(focus = 'acc')
# get_table(focus = 'fair')
get_table(focus = f'auto_tol{tol}')
# print(result_dict)

# ###########################################################################################


# '''
# testing the sensitivity of label budget r: args.new_data_each_round
# '''
# evaluate_impact_of_label_budget = False
# evaluate_impact_of_label_budget_val = False



# if evaluate_impact_of_label_budget == True:

#     sel_layers = [4]
#     # default strategy: [0, 1, 6, 2, 5]
#     # select BALD as the baseline to compare
#     strategy = [6, 5]
    
#     if dataset == 'celeba':
#         # label_key = ['Smiling', 'Straight_Hair', 'Attractive', 'Pale_Skin', 'Young', 'Big_Nose']
#         label_key = [ 'Smiling', 'Attractive']#'Smiling' 'Attractive', 'Young', 'Big_Nose'
#         # label_key = ['Young']
#         # label_key = ['Attractive']
#     elif dataset == 'compas':
#         label_key = ['label']
#     elif dataset == 'adult':
#         label_key = ['label']
#     elif dataset == 'jigsaw':
#         label_key = ['label']
#     else:
#         raise NameError(f'undefined dataset {dataset}')
#     # take one fairness definition to compare 
#     metrics = ['dp']

# else: 
    
#     #By default, we compare the results
#     sel_layers = [4]
#     # default strategy: [0, 1, 6, 2, 5]
#     strategy = [7]
#     if dataset == 'celeba':
#         # label_key = ['Smiling', 'Straight_Hair', 'Attractive', 'Pale_Skin', 'Young', 'Big_Nose']
#         label_key = [ 'Smiling']#'Smiling' 'Attractive', 'Young', 'Big_Nose'
#         # label_key = ['Young']
#         # label_key = ['Attractive']
#         label_tag = '1024'

#     elif dataset == 'compas':
#         label_key = ['label']
#         label_tag = 'budget'


#     elif dataset == 'adult':
#         label_key = ['label']
#         label_tag = 'budget'

#     elif dataset == 'jigsaw':
#         label_key = ['label']
#         label_tag = '512'
#     else:
#         raise NameError(f'undefined dataset {dataset}')
#     # metrics = ['dp', 'eop', 'eod']
#     metrics = ['dp']

# # '''
# see the impact of label budgets 
# '''

# if evaluate_impact_of_label_budget == True:
#     # read logs, then save the processed results to dict
#     for layer in sel_layers:
#         for stg in strategy:
#             for label in label_key:
#                 for metric in metrics:

#                     # if stg == 1:
#                     #     file_name = f'{label}_s{stg}_{metric}_2'
#                     # else:
#                     if stg > 0:
#                         if evaluate_impact_of_label_budget_val == True:
#                             file_name = f'{label}_s{stg}_{metric}_{layer}_' + label_tag 
#                         else:
#                             file_name = f'{label}_s{stg}_{metric}_{layer}'  
#                         if 'res18' in root:
#                             file_name = 'res18_' + file_name
#                         if suffix:
#                             file_name += suffix

#                         result_dict_val[file_name] = []
#                         # get the list of validation dataset to plot figures
#                         if evaluate_impact_of_label_budget_val == True:
                            
#                             val_length_of_file = get_val_length(file_name)
#                             val_lengths = list(range(val_length_of_file))
#                             print('###############val_length_of_file: ' + str(val_length_of_file))
#                             for val_length in val_lengths:

#                                 result_current_val_length = get_result_val(file_name, val_length)
#                                 print(file_name, result_current_val_length, val_length)
#                                 result_dict_val[file_name].append(result_current_val_length)

#                         # normal table data
#                         else:
#                                 base_pref = get_result_typical(file_name)
#                                 print(file_name, base_pref)
#                                 # import pdb
#                                 # pdb.set_trace()
#                                 # if evaluate_impact_of_label_budget == True:
#                                 #     result_dict[f'{label}_s0_{metric}_{layer}_{label_budget}' + suffix] = [base_pref]
#                                 # else:
#                                 #     result_dict[f'{label}_s0_{metric}_{layer}' + suffix] = [base_pref]
#                     else: 
#                         result_dict[f'{label}_s0_{metric}_{layer}' + suffix] = [base_pref]


#     if evaluate_impact_of_label_budget_val == True:
#         print("################# result_dict_val:  " + str(result_dict_val))

#     else:
#         print("################# result_dict:  " + str(result_dict))
#     # get table
#     # get_table(focus = 'acc')
#     # get_table(focus = 'fair')

#     # get_table(focus = f'auto_tol{tol}')



#     #print(result_dict)


#     # 遍历字典中的每个键和值


#     # 遍历字典中的每个键和对应的列表
#     for key, value_list in result_dict_val.items():
#         # 过滤掉列表中的nan值
#         filtered_list = [item for item in value_list if not np.isnan(item[0]) and not np.isnan(item[1])]

#         # 以键名创建文件名
#         filename = f"{key}.txt"
#         print('file_name::  ' + filename)

#         # 将过滤后的列表写入文件
#         with open(root + filename, 'w') as file:
#             for item in filtered_list:
#                 file.write(f'{item}\n')


# ###########################################################################################