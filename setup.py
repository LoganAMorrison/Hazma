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
      author='Logan Morrison and Adam Coogan',
      author_email='loanmorr@ucsc.edu',
      maintainer='Logan Morrison',
      maintainer_email='loanmorr@ucsc.edu',
      url='http://hazma.readthedocs.io/en/latest/',
      description='Package for computing FSR and decay spectra for light \
      particles',
      long_description="""Package for computing the FSR and decay spectra for
      light mesons (pions and kaons) and light fermions (electron and muon).
      """,
      packages=packs,
      ext_modules=cythonize(extensions),
      include_dirs=[np.get_include(), 'hazma/decay_helper_functions'],
      include_package_data=True,
      license='MIT License',
      platforms='MacOS and Linux',
      download_url='https://github.com/LoganAMorrison/Hazma'
      )
