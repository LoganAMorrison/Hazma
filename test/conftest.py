import pathlib

THIS_DIR = pathlib.Path(__file__).parent.absolute()
PKG_DIR = THIS_DIR.joinpath("..")

# test/decay/ was removed with hazma/_decay/ (cython-to-rust Task 0.3), and
# test_gamma_ray.py with hazma/gamma_ray.py (Task 0.2, ADR-0003). Nothing in
# test/ is skipped at collection any more; setup.py is not a test module.
collect_ignore = [PKG_DIR.joinpath("setup.py")]
