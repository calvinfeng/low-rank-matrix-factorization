import lowrank
import numpy as np


DATA_DIR = 'datasets/100k/'
def main():
    converter = lowrank.MatrixConverter(movies_filepath=DATA_DIR + 'movies.csv',
                                        ratings_filepath=DATA_DIR + 'ratings.csv')
    training_rating_mat, test_rating_mat = converter.get_rating_matrices()

    factorizer = lowrank.Factorizer(training_rating_mat, test_rating_mat, feature_dim=10, reg=0.05)
    benchmarks = factorizer.train(steps=400, learning_rate=1e-4)

    rmses = [bm[2] for bm in benchmarks]
    print rmses

    lowrank.unload_movie_features(filepath=DATA_DIR + 'features.csv',
                                  feature_dim=10,
                                  movie_features=converter.get_movie_feature_map(factorizer.M))

if __name__ == '__main__':
    main()
