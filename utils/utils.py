import yaml
import pickle

class Logger(object):
    def __init__(self, file):
        self.file = file
    def save_log(self, text):
        print(text)
        with open(self.file, 'a') as f:
            f.write(text + '\n')



def read_yaml(path):
    """
    Read a yaml file from a certain path.
    """
    stream = open(path, 'r')
    dictionary = yaml.safe_load(stream)
    return dictionary


def write_pickle(path,data):

    with open(path,'wb') as file:
        pickle.dump(data,file)