import lowrank
import numpy as np


DATA_DIR = 'datasets/100k/'
def main():
    converter = lowrank.MatrixConverter(movies_filepath=DATA_DIR + 'movies.csv',
                                        ratings_filepath=DATA_DIR + 'ratings.csv')
    training_rating_mat, test_rating_mat = converter.get_rating_matrices()

    factorizer = lowrank.Factorizer(training_rating_mat, test_rating_mat)
    grad_u, grad_m = factorizer.gradients()
    print 'Done with analytical gradients'
    num_grad_u = factorizer.num_gradients()
    print 'Done with numerical gradients'

    print np.matrix.round(grad_u - num_grad_u, decimals=5)
    # print np.matrix.round(grad_m - num_grad_m, decimals=5)


if __name__ == '__main__':
    main()
