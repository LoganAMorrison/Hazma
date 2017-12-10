from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

packs = ['hazma',
         'hazma.helper_functions',
         'hazma.particles',
         'hazma.rambo']

extensions = ["hazma/particles/*.pyx", "hazma/rambo/*.pyx"]

setup(name='hazma',
      version='1.0',
      description='Gamma Ray Spectrum Generator',
      author='Logan Morrison and Adam Coogan',
      author_email='loanmorr@ucsc.edu',
      url='',
      packages=packs,
      ext_modules=cythonize(extensions),
      include_dirs=[np.get_include(), 'hazma/particles']
      )
