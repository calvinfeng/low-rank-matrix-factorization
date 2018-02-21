import lowrank
import numpy as np


DATA_DIR = 'datasets/100k/'
def main():
    converter = lowrank.MatrixConverter(movies_filepath=DATA_DIR + 'movies.csv',
                                        ratings_filepath=DATA_DIR + 'ratings.csv')
    training_rating_mat, test_rating_mat = converter.get_rating_matrices()

    factorizer = lowrank.Factorizer(training_rating_mat, test_rating_mat)
    factorizer.train()


if __name__ == '__main__':
    main()
