import numpy as np

from sklearn.tree import DecisionTreeRegressor

class ModifiedTree():
    """
    Objects of this class contain a regression tree from sklearn and
    give the possibility to change the values of the leaves 
    """
    def __init__(self, max_depth=2, min_samples_leaf=2):
        self.max_depth = max_depth
        self.tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.values = None
    
    def fit(self, X, R):
        """Fit the tree on the residuals R"""
        if self.max_depth > 0:
            self.tree.fit(X, R)

    def updateLeaves(self, X, d_l, d2_l):
        """
        Update the values of all leaves using the inputs X of shape (n, d)
        and the clipped gradients xi of shape (n, )
        """
        if self.max_depth == 0:
            xi = - np.sum(d_l) / np.sum(d2_l)
            self.values = [np.clip(xi, -1, 1)]

        else:
            # Get the indices of the leaves for all datapoints
            leaf_indices = self.tree.apply(X) 
            values = []
            for i in range(max(leaf_indices)+1):
                # Create a boolean array to represent the datapoints in leaf i
                indices = leaf_indices==i
                if np.sum(indices) > 0:
                    xi = - np.sum(d_l*indices) / np.sum(d2_l*indices)
                    values.append(np.clip(xi, -1, 1))

                else:
                    values.append(0)

            self.values = np.array(values)

    def predict(self, X):
        """Predicts outputs using the modified values of the leaves"""
        if self.max_depth == 0:
            if self.values is not None:
                return np.ones(X.shape[0]) * self.values[0]        

        if self.values is None:
            return self.tree.predict(X)

        node_predictions = self.tree.apply(X)
        predictions = self.values[node_predictions]

        return predictions    