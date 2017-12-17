from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

packs = ['hazma',
         'hazma.fsr_helper_functions',
         'hazma.decay_helper_functions',
         'hazma.phases_space_generator']

extensions = ["hazma/decay_helper_functions/*.pyx",
              "hazma/phases_space_generator/*.pyx"]

setup(name='hazma',
      version='1.0',
      description='Gamma Ray Spectrum Generator',
      author='Logan Morrison and Adam Coogan',
      author_email='loanmorr@ucsc.edu',
      url='',
      packages=packs,
      ext_modules=cythonize(extensions),
      include_dirs=[np.get_include(), 'hazma/decay_helper_functions'],
      include_package_data=True
      )
