November 23rd (Logan)
---------------------
There are a bunch of cases of `dict.iteritems` in hazma. `iteritems` was removed in python 3. In python 3`dict.items()` is equivalent to `dict.items()`. In python 2, `dict.items()` makes a list of tuples, which is inefficient. To make things compatible with python 3, I am changing `dict.iteritems` to `dict.items()`. I did so in: 
- `hazma.theory._theory_cmb.py` (ln 127)
- `hazma.theory._theory_constrain.py` (lns 46, 103, 143, 157)
- `hazma.theory._theory_gamma_ray_limits.py` (ln 104)
- `scalar_mediator._scalar_mediator_contraints.py` (ln 245)

November 27th (Logan)
---------------------
Adding test classes. First bug with vector mediator spectrum when the vector has zero decay width (ln 77) in `hazma._vector_mediator_spectra.py`.
