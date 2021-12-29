class Node:
    def __init__(self, label = None, children = None, a_star = None):
        self.label = label
        self.children = {} if children is None else children
        self.a_star = a_star

    # you may want to add additional fields here...
