import pathlib

THIS_DIR = pathlib.Path(__file__).parent.absolute()
PKG_DIR = THIS_DIR.joinpath("..")

collect_ignore = [PKG_DIR.joinpath("setup.py")]

# test/decay/ was removed with hazma/_decay/ (cython-to-rust Task 0.3); the
# only entry left here is test_gamma_ray.py, which exercises the
# broken-on-import hazma.gamma_ray module.
old_tests_ignore = [
    THIS_DIR.joinpath("test_gamma_ray.py"),
]


collect_ignore.extend(old_tests_ignore)
