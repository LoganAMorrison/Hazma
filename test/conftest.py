# test/decay/ was removed with hazma/_decay/ (cython-to-rust Task 0.3), and
# test_gamma_ray.py with hazma/gamma_ray.py (Task 0.2, ADR-0003). The last
# entry was the repo's setup.py, which the maturin cutover deleted
# (Task 7.1). Nothing is skipped at collection any more, and this empty
# list is the statement of that -- a `collect_ignore` naming a path that no
# longer exists silently protects nothing.
collect_ignore: list[str] = []
