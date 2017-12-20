from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

# 'hazma.gamma_ray_helper_functions'
# "hazma/gamma_ray_helper_functions/*.pyx"

packs = ["hazma",
         "hazma.fsr_helper_functions",
         "hazma.decay_helper_functions",
         "hazma.phase_space_generator",
         "hazma.gamma_ray_helper_functions"]

extensions = [Extension("*", ["hazma/decay_helper_functions/*.pyx"]),
              Extension("*", ["hazma/phase_space_generator/*.pyx"]),
              Extension("*", ["hazma/gamma_ray_helper_functions/*.pyx"])]

setup(name='hazma',
      version='1.1',
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
