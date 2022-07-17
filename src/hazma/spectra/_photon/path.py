from importlib.resources import path
import pathlib

THIS_DIR = pathlib.Path(__file__).absolute().parent
DATA_DIR = THIS_DIR.joinpath("data")
