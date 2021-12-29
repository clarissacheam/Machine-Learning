
import parse
import ID3
import random


def GetLearningCurve(inFile):
    average_without_pruning = []
    average_with_pruning = []
    training_size = []

    data = parse.parse(inFile)

    for s in range(10, 300, 10):
        training_size.append(s)
        without_prune_test_accuracy_100 = []
        with_prune_test_accuracy_100 = []
        x = int(0.85 * s)

        for i in range(100):
            random.shuffle(data)
            trainset = data[:s]    # training set
            train = trainset[:x]   # training set  for prune: 85% of training set
            valid = trainset[x:]   # validation set for prune: 15% of training set
            test = data[s:]        # data set without training set

            without_prune_tree = ID3.ID3(trainset, 'democrat')
            without_prune_test_accuracy = ID3.test(without_prune_tree, test)
            without_prune_test_accuracy_100.append(without_prune_test_accuracy)

            with_prune_tree = ID3.ID3(train, 'democrat')
            ID3.prune(with_prune_tree, valid)
            with_prune_test_accuracy = ID3.test(with_prune_tree, test)
            with_prune_test_accuracy_100.append(with_prune_test_accuracy)

        average_without_pruning.append(sum(without_prune_test_accuracy_100) / len(without_prune_test_accuracy_100))
        average_with_pruning.append(sum(with_prune_test_accuracy_100) / len(with_prune_test_accuracy_100))

    return average_without_pruning, average_with_pruning, training_size

