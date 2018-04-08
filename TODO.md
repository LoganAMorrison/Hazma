TODO
====

- [x] Check Derivation of AP
- [ ] Make Function Arguments more Intuitive and Consistent
- [ ] Make Function Documentations
- [ ] Negative Squared Matrix Elements

Check Derivation of AP (FIXED)
------------------------------

The AP results seem to differ from ours by a factor of two. We should look at
the derivation of the AP result. - Logan

In FeynCalc I was forgetting to change Lorentz indices when computing the
complex conjugates of amplitudes. This resulted in a factor of two. - Logan

Make Function Arguments more Intuitive and Consistent
-----------------------------------------------------

I feel like many of the functions have differing argument structures. They should be modified to have a more intuitive structure. Also, some functions which are similar or in same module have very different argument structure. This could be confusing for users. - Logan

Make Function Documentations
----------------------------

Not all functions have documentation. - Logan


Negative Squared Matrix Elements
--------------------------------
I've noticed that some squared matrix elements involving mesons are negative.
For example, the decay $K_{L}\to$ and $\chi\chi\to\pi^{+}\pi^{-}\gamma$ - Logan
