import numpy as np

from ModifiedTree import ModifiedTree

# Random Forest quantile estimator
class RF_Quantile():
    """
    Random forest estimator of the quantiles
    """
    def __init__(self, tau_0, N_trees=100, D_max=5, L_min=10, s=0.1):
        # Level of the quantiles to estimate
        self.tau_0 = tau_0

        # Number of trees 
        self.N_trees = N_trees 

        # Maximum depth of the trees
        self.D_max = D_max

        # Minimum number of datapoints per leaf
        self.L_min = L_min

        # Fraction of the dataset used for each tree
        self.s = s

        # Lists of regression trees
        self.trees = []


    def predict(self, X):
        """Use the lists of trees to predict the quantiles for the inputs X, whose shape is (n, d)"""
        q_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            q_pred += tree.predict(X)
        q_pred /= self.N_trees
        return q_pred


    def fit(self, X, Y):
        """Fit the Random forest"""
        # Reset the estimators
        self.trees = []

        n = X.shape[0]
        for t in range(self.N_trees):
            # Step 1: draw a subsample
            S_b = np.random.choice(np.arange(n), int(self.s*n), replace=False)
            X_b = X[S_b]
            Y_b = Y[S_b]

            self.trees.append(self.get_tree(X_b, Y_b))
            

    def get_tree(self, X, Y):
        """Perform steps 2, 3 and 4 of the algorithm 1"""

        tree = ModifiedTree(max_depth=self.D_max, min_samples_leaf=self.L_min)

        tree.fit(X, Y)

        # Get the indices of the leaves for all datapoints
        leaf_indices = tree.tree.apply(X) 
        values = []
        for i in range(max(leaf_indices)+1):
            # Create a boolean array to represent the datapoints in leaf i
            indices = leaf_indices==i
            if np.sum(indices) > 0:
                distribution = Y[indices]
                values.append(np.quantile(distribution, self.tau_0))

            else:
                values.append(0)

        tree.values = np.array(values)

        return tree
