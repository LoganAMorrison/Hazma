import os


def get_dir_path():
    this_dir, this_filename = os.path.split(__file__)
    return this_dir
