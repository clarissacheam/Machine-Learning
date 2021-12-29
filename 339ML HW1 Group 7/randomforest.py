import parse
import random
from collections import Counter
import ID3


class RandomForestClassifier:
    def __init__(self):
        self.tree_collection = []


def evaluate(forest, example):
    anslist = {}
    for tree in forest.tree_collection:
        ans = ID3.evaluate(tree, example)
        if ans not in list(anslist.keys()):
            anslist[ans] = 1
        else:
            anslist[ans] += 1
    c = Counter(anslist)
    
    return c.most_common()[0][0]


def test(forest, examples):
    correct = 0
    total = 0

    for i in examples:
        total += 1
        class_label = evaluate(forest, i)
        if class_label == i['Class']:
            correct += 1

    return float(correct) / float(total)


def train(examples, forest_size):
    forest = RandomForestClassifier()

    # Using the simplest form of bootstrap aggregating method, get sub dataset
    ds_collection = bagging(examples, forest_size)
    for ds in ds_collection:
        tree = ID3.ID3(ds, 0)
        if tree is not None:
            forest.tree_collection.append(tree)
    
    return forest


def bagging(examples, forest_size):
    dataset_collection = []
    sz = 4*len(examples)//5
    for i in range(0, forest_size):
        random.shuffle(examples)
        ds = examples[:sz]
        dataset_collection.append(ds)
    return dataset_collection


candy_dataset = parse.parse('candy.data')

r_accuracy_sum = 0
t_accuracy_sum = 0
iteration = 100

for i in range(iteration):
    random.shuffle(candy_dataset)
    candy_train_dataset = candy_dataset[:3*len(candy_dataset)//4]
    candy_test_dataset = candy_dataset[3*len(candy_dataset)//4:]

    # create a forest size 350
    r_forest = train(candy_train_dataset, 350)
    r_accuracy_sum += test(r_forest, candy_test_dataset)

    tree = ID3.ID3(candy_train_dataset, 0)
    t_accuracy_sum += ID3.test(tree, candy_test_dataset)

print('Average random forest accuracy (forest size: 350) over', iteration, 'iteration ->', r_accuracy_sum/iteration)
print('Average single ID3 accuracy over', iteration, 'iteration ->', t_accuracy_sum/iteration)