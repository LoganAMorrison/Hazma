from importlib.resources import path
import pathlib

THIS_DIR = pathlib.Path(__file__).parent.absolute()
PKG_DIR = THIS_DIR.joinpath("..")

collect_ignore = [PKG_DIR.joinpath("setup.py")]

old_tests_ignore = [
    THIS_DIR.joinpath("test_gamma_ray.py"),
    *THIS_DIR.joinpath("decay").iterdir(),
]


collect_ignore.extend(old_tests_ignore)
