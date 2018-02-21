import numpy as np


class Factorizer(object):
    def __init__(self, training_rating_matrix, test_rating_matrix, reg=0.0, learning_rate=1e-5, feature_dim=10):
        """Instantiate the necessary matrices to perform low rank factorization

        :param training_rating_matrix numpy.array: A sparse numpy matrix that holds ratings from every user on every movie.
        :param test_rating_matrix numpy.array: A sparse numpy matrix that holds ratings from every user on every movie.
        :param reg float: Regularization strength
        :param learninG_rate float: Learning rate for gradient descent update rule
        :param feature_dim int: Dimension for the latent vector, i.e. feature for movie, preference for user
        """
        if training_rating_matrix.shape != test_rating_matrix.shape:
            raise "dimension of training set does not match up with that of test set"

        self.reg = reg
        self.learning_rate = learning_rate
        self.feature_dim = feature_dim
        self.user_dim, self.movie_dim = training_rating_matrix.shape

        # Instantiate the low rank factorizations
        self.U = np.random.rand(self.user_dim, self.feature_dim)
        self.M = np.random.rand(self.movie_dim, self.feature_dim)
        self.R = training_rating_matrix
        self.R_test = test_rating_matrix

    def loss(self):
        """Computes L2 loss
        """
        pred_R = np.dot(self.U, self.M.T)

        loss = 0
        itr = np.nditer(self.R, flags=['multi_index'])  # itr stands for iterator
        while not itr.finished:
            if self.R[itr.multi_index] != 0:
                diff = self.R[itr.multi_index] - pred_R[itr.multi_index]
                loss += 0.5 * diff * diff
            itr.iternext()

        # Factor in regularizations
        loss += self.reg * np.sum(self.U * self.U) / 2
        loss += self.reg * np.sum(self.M * self.M) / 2

        return loss

    def gradients(self):
        grad_R = np.dot(self.U, self.M.T) - self.R

        itr = np.nditer(grad_R, flags=['multi_index'])
        while not itr.finished:
            if self.R[itr.multi_index] == 0:
                grad_R[itr.multi_index] = 0
            itr.iternext()

        grad_u = np.dot(grad_R, self.M) + (self.reg * self.U)
        grad_m = np.dot(grad_R.T, self.U) + (self.reg * self.M)

        return grad_u, grad_m

    def num_gradients(self, h=1e-5):
        """Compute numerical gradients for U and M
        """
        num_grad_u = np.zeros(self.U.shape)
        num_grad_m = np.zeros(self.M.shape)

        U_dim, L_dim = self.U.shape
        M_dim, L_dim = self.M.shape

        progress = 0
        itr = np.nditer(num_grad_u, flags=['multi_index'])
        while not itr.finished:
            indices = itr.multi_index

            # Store the old value
            old_val = self.U[indices]

            self.U[indices] = old_val + h
            fuph = self.loss()

            self.U[indices] = old_val - h
            fumh = self.loss()

            self.U[indices] = old_val
            num_grad_u[indices] = (fuph - fumh) / (2 * h)

            progress += 1
            print progress
            itr.iternext()

        return num_grad_u

        itr = np.nditer(num_grad_m, flags=['multi_index'])
        while not itr.finished:
            indices = itr.multi_index

            # Store the old value
            old_val = self.M[indices]

            self.M[indices] = old_val + h
            fmph = self.loss()

            self.M[indices] = old_val - h
            fmmh = self.loss()

            self.M[indices] = old_val
            num_grad_m[indices] = (fmph - fmmh) / (2 * h)

            itr.iternext()

        return num_grad_u, num_grad_m
