import pickle


def save_pickle(file, path):
    pickle.dump(file, open(path, "wb"))


def load_pickle(path):
    return pickle.load(open(path, "rb"))
