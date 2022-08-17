from typing import List

class DecisionTreeClassifier:
    def __init__(self, max_depth: int):
        self.tree = None
        self.max_depth = max_depth
        self.min_size = 5

    def fit(self, X: List[List[float]], y: List[int]):
        train = X
        for i in range(len(X)):
            train[i].append(y[i])

        self.tree = self.__build_tree(train, self.max_depth, self.min_size)
        return self.tree

    def predict(self, X: List[List[float]]):
        predictions = list()
        for row in X:
            prediction = self.__get_prediction(self.tree, row)
            predictions.append(prediction)
        return(predictions)

    # Split a dataset based on an attribute and an attribute value
    def __test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Calculate the Gini index for a split dataset
    def __gini_index(self, groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Select the best split point for a dataset
    def __get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.__test_split(index, row[index], dataset)
                gini = self.__gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    # Create a terminal node value
    def __to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # Create child splits for a node or make terminal
    def __split(self, node, max_depth, min_size, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.__to_terminal(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.__to_terminal(left), self.__to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.__to_terminal(left)
        else:
            node['left'] = self.__get_split(left)
            self.__split(node['left'], max_depth, min_size, depth+1)
        # process right child
        if len(right) <= min_size:
            node['right'] = self.__to_terminal(right)
        else:
            node['right'] = self.__get_split(right)
            self.__split(node['right'], max_depth, min_size, depth+1)

    # Build a decision tree
    def __build_tree(self, train, max_depth, min_size):
        root = self.__get_split(train)
        self.__split(root, max_depth, min_size, 1)
        return root

    # Get a prediction with a decision tree
    def __get_prediction(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.__get_prediction(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.__get_prediction(node['right'], row)
            else:
                return node['right']