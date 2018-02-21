import csv


def load_movies(filepath):
    """Returns a dictionary of movie ID mapping to its relevant information and attributes

    :param string filepath:
    """
    movies = dict()
    with open(filepath, 'rb') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader, None)  # Skip header
        for row in reader:
            movie_id, title, _ = row
            movies[int(movie_id)] = {"id": int(movie_id), "title": title}

    return movies


def load_user_ratings(filepath, max_num_user=10000):
    """Returns a dictionary of user ID mapping to ratings submitted by the user

    :param string filepath:
    :param int max_num_user:
    """
    user_ratings = dict()
    with open(filepath, 'rb') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader, None)  # Skip header
        for row in reader:
            user_id, movie_id, rating = int(row[0]), int(row[1]), float(row[2])
            if user_ratings.get(user_id) is None:
                user_ratings[user_id] = dict()

            user_ratings[user_id][movie_id] = rating

    reduced_user_ratings = dict()
    for user_id in user_ratings:
        if len(user_ratings[user_id]) > 300:
            reduced_user_ratings[user_id] = user_ratings[user_id]

        if len(reduced_user_ratings) == max_num_user:
            return reduced_user_ratings

    return reduced_user_ratings
