import numpy as np
import math


def get_entropy(prob):
    if prob == 1 or prob == 0:
        return 0 
    else:
        return (-1*prob*math.log2(prob))-((1-prob)*math.log2(1-prob))


def info_gained(left, right):
    parent = left + right

    if len(parent) > 0: 
        parent_prob = parent.count(1)/ len(parent)
    else:
        parent_prob = 0

    if len(left) > 0:
        left_prob = left.count(1) / len(left)
    else:
        left_prob = 0

    if len(right) > 0:
        right_prob = right.count(1) / len(right)
    else:
        right_prob = 0 

    info_parent = get_entropy(parent_prob)
    info_left = get_entropy(left_prob)
    info_right = get_entropy(right_prob)

    total_info_gained = info_parent - ((len(left)/len(parent))*info_left)-((len(right)/len(parent))*info_right) 

    return total_info_gained


def create_bootstrap(x_train, y_train):
    boot_indxs = list(np.random.choice(range(len(x_train), len(x_train), replace=True)))
    out_of_bound_indxs = [i for i in range(len(x_train)) if i not in boot_indxs]

    x_boot = x_train.iloc[boot_indxs].values
    y_boot = y_train.iloc[boot_indxs]
    x_out_of_bound = x_train.iloc[x_out_of_bound].values
    y_out_of_bound = y_train.iloc[y_out_of_bound]

    return x_boot, y_boot, x_out_of_bound, y_out_of_bound
    

def get_out_of_bag_score(tree, x_test, y_test):
    wrong = 0 

    for i in range(len(x_test)):
        pred = predict_tree(tree, x_test[i])
        if pred != y_test[i]:
            wrong+=1
    
    return wrong/len(x_test)


def find_split(x_boot, y_boot, max_num_feature):
    features - list()
    num_features = len(x_boot[0])

    while len(features) <= max_num_feature:
        feature_idx = random.sample(range(num_features), 1)
        if feature_idx not in features:
            features.extend(feature_idx)

    best_info_gained = -2 

    for feature_idx in features:
        for split_point in x_boot[:,feature_idx]:
            left_child = {'x_boot': [], 'y_boot': []}
            right_child= {'x_boot': [], 'y_boot': []}

            if type(split_point) in [int,float]:
                for i, value in enumerate(x_boot[:,feature_idx]):
                    if value<=split_point:
                        left_child['x_boot'].append(x_boot[i])
                        left_child['y_boot'].append(y_boot[i])
                    else:
                        right_child['x_boot'].append(x_boot[i])
                        right_child['y_boot'].append(y_boot[i])
            else:
                for i, value in enumerate(x_boot[:,feature_idx]):
                    if value == split_point:
                        left_child['x_boot'].append(x_boot[i])
                        left_child['y_boot'].append(y_boot[i])
                    else:
                        right_child['x_boot'].append(x_boot[i])
                        right_child['y_boot'].append(y_boot[i])

            split_ig = info_gained(left_child['y_boot'], right_child['y_boot'])

            if split_ig > best_info_gained:
                best_info_gained = split_ig
                left_child['x_boot'] = np.array(left_child['x_boot'])
                right_child['x_boot'] = np.array(right_child['x_boot'])
                node = {'information_gain': split_info_gain,
                    'left_child': left_child,
                    'right_child': right_child,
                    'split_point': split_point,
                    'feature_idx': feature_idx}

    return node


def terminal_node(node):
    y_boot = node['y_boot']
    pred = max(y_boot, key = y_boot.count)
    return pred


def split_node(node, max_features, min_samples_split, max_depth, depth):
    left_child = node['left_child']
    right_child = node['right_child']

    del(node['left_child'])
    del(node['right_child'])

    if len(left_child['y_boot']) == 0 or len(right_child['y_boot']) == 0:
        empty = {'y_boot': right_child['y_boot']+ left_child['y_boot']}
        node['right_split'] = node['left_split'] = terminal_node(empty)
        return

    if depth >= max_depth:
        node['left_split'] = terminal_node(left_child)
        node['right_split'] = terminal_node(right_child)
        return node
    
    if len(left_child['x_boot']) <= min_samples_split:
        node['left_split'] = node['right_split'] = terminal_node(left_child)
    else:
        node['left_split'] = find_split(left_child['x_boot'], left_child['y_boot'], max_features)
        split_node(node['left_split'], max_features, min_samples_split, max_depth, depth + 1)

    if len(right_child['x_boot']) <= min_samples_split:
        node['left_split'] = node['right_split'] = terminal_node(right_child)
    else:
        node['right_split'] = find_split(right_child['x_boot'], right_child['y_boot'], max_features)
        split_node(node['right_split'], max_features, min_samples_split, max_depth, depth+1)


def build_tree(x_boot, y_boot, max_depth, min_samples_split, max_features):
    root = find_split(x_boot, y_boot, max_features)
    split_node(root, max_features, min_samples_split, max_depth, 1)

    return root


def random_forest(x_train, y_train, n_estimators, max_features, max_depth, min_samples_split):
    tree_list = list()
    out_of_bound_list = list()

    for i in range(n_estimators):
        x_boot, y_boot, x_out_of_bound, y_out_of_bound = create_bootstrap(x_train, y_train)

        tree = build_tree(x_boot, y_boot, max_depth, min_samples_split, max_features)
        tree_list.append(tree)

        out_of_bound_list.append(get_out_of_bag_score(tree, x_out_of_bound, y_out_of_bound)) 

    print("Out Of Bag estimate: {:.2f}".format(np.mean(out_of_bound_list)))
    return tree_list


def predict_tree(tree, x_test):
    feature_idx = tree['feature_idx']

    if x_test[feature_idx] <= tree['split_point']:
        if type(tree['left_split']) == dict:
            return predict_tree(tree['left_split'], x_test)
        else:
            return tree['left_split']
    else:
        if type(tree['right_split']) == dict:
            return predict_tree(tree['right_split'], x_test)
        else:
            return tree['right_split']


def predict_random_forest(tree_list, x_test):
    pred_list = list()

    for i in range(len(x_test)):
        test_pred = [predict_tree(tree, x_test.values[i]) for tree in tree_list]
        final_pred = max(test_pred, key = test_pred.count)
        pred_list.append(final_pred)
    
    return np.array(pred_list)




    

    

print(get_entropy(1))

