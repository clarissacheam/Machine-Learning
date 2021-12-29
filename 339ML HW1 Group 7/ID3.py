from node import Node
import math
import parse
from collections import Counter


def ID3(examples, default):
    '''
    Takes in an array of examples, and returns a tree (an instance of Node)
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    '''  

    root = Node(label=default)
    # Label with the most common class
    c = Counter(x['Class'] for x in examples)
    root.label = c.most_common()[0][0]
    sub = {}
    # if only 1 example left
    if len(examples) == 1:
        return root
    # all examples share the same label, return the node
    if all(examples[0]['Class'] == x['Class'] for x in examples):
        return root
    # attribute is empty, return the node
    if len(examples[0].keys()) == 1:
        return root
        
    # find the best attribute A*
    ig_max = float('-inf')
    A_star = ''
    potential_splits = {}
    # calculate the current entropy before splitting
    curr_entropy = get_entropy(examples)
    attrs = list(examples[0].keys())
    attrs.remove('Class')
    for attr in attrs:
        # calculate the entropy of each attribute
        value_unique = Counter(x[attr] for x in examples).keys()
        value_unique_count = Counter(x[attr] for x in examples).values()
        sub_entropy = 0
        for value in value_unique:
            subnodes = [x for x in examples if x[attr] == value]
            potential_splits[attr] = len(subnodes)
            sub_entropy += len(subnodes)/len(examples)*get_entropy(subnodes)
        ig = curr_entropy - sub_entropy
        if ig > ig_max:
            ig_max = ig
            A_star = attr
        
    # assign values and new tree
    root.a_star = A_star

    for value in Counter(x[A_star] for x in examples).keys():
        child_tree = ID3(split(examples, A_star, value), c.most_common()[0][0])
        # set subtree to dictionary key v
        sub.update({value: child_tree})
        # set the sub as children
        if isinstance(sub, dict):
            root.children = sub

    return root


def prune(node, examples):
    '''
      Takes in a trained tree and a validation set of examples.  Prunes nodes in order
      to improve accuracy on the validation data; the precise pruning strategy is up to you.
    '''

    # if leaf, skip
    def isleaf(node):
        return (len(node.children) == 0)

    def inner_prune(node, d_tree, parent_node, c_key, examples):
        if isleaf(node):
            return

        # we keep on going deep till bottom of the tree
        for c in list(node.children.keys()):
            if not isleaf(node.children[c]):
                inner_prune(node.children[c], d_tree, node, c, examples)

        # all children must be leaves in order to prune the node
        for c in list(node.children.keys()):
            # if child not leaf, skip pruning
            if not isleaf(node.children[c]):
                return

        # if the code reach here, we are good to try prune
        # compute accuracy before pruning
        accuracy_before = test(d_tree, examples)

        # make a temp leaf node to replace prune node
        temp_leaf_node = Node(parent_node.children[c_key].label)
        parent_node.children[c_key] = temp_leaf_node

        # compute accuracy after prune
        accuracy_after = test(d_tree, examples)
        # if the pruned accuracy is worse then before prune, replace back the temp leaf with the pruned node
        if accuracy_after < accuracy_before:
            parent_node.children[c_key] = node

        return

    # if leaf, skip
    if isleaf(node):
        return

    # for every child branch, prune
    for key in node.children:
        inner_prune(node.children[key], node, node, key, examples)

    return


def test(node, examples):
    '''
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    '''
    mis = 0
    for example in examples:
        predict = evaluate(node, example)
        if predict != example['Class']:
            mis += 1
    # calculate accuracy
    acc = 1 - (float(mis) / len(examples))
    return acc


def evaluate(node, example):
    '''
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    '''
    curr = node
    while curr.children != {}:
        temp_c = curr.children
        if curr.children.get(example[curr.a_star], None):
            curr = curr.children[example[curr.a_star]]
        else:
            curr = temp_c[list(temp_c.keys())[0]]
    return curr.label


def get_entropy(examples):
    # if there's only one example passed in
    try:
        all(examples[0]['Class'] == x['Class'] for x in examples)
    except KeyError:
        return 0
    if all(examples[0]['Class'] == x['Class'] for x in examples):
        return 0
    else:
        classes_unique_count = Counter(x['Class'] for x in examples).values()
        entropy = 0
        entropy += sum(-float(x)/len(examples)*math.log(float(x)/len(examples),2) for x in classes_unique_count)
    return entropy

def split(examples, attrib, attrib_val):
    # remove best attribute from all elements of examples
    result = []

    for e in examples:
        if e[attrib] == attrib_val:
            copy = e.copy()
            del copy[attrib]
            result.append(copy)

    return result
