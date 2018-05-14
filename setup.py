from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize

hdhf = "hazma/decay_helper_functions"
hgrhf = "hazma/gamma_ray_helper_functions"
hpshf = "hazma/phase_space_helper_functions"
hfthf = "hazma/field_theory_helper_functions"
hphf = "hazma/positron_helper_functions"

packs = ["hazma",
         "hazma.axial_vector_mediator",
         "hazma.decay_helper_functions",
         "hazma.field_theory_helper_functions",
         "hazma.gamma_ray_helper_functions",
         "hazma.gamma_ray_limits",
         "hazma.phase_space_helper_functions",
         "hazma.positron_helper_functions",
         "hazma.pseudo_scalar_mediator",
         "hazma.scalar_mediator",
         "hazma.unitarization",
         "hazma.vector_mediator"]

decay_ext = Extension("*", sources=[hdhf + "/*.pyx"])
gamma_ext = Extension("*", sources=[hgrhf + "/*.pyx"])
phase_ext = Extension("*", sources=[hpshf + "/*.pyx"],
                      extra_compile_args=['-g', '-std=c++11'],
                      language="c++")
field_theory_ext = Extension("*", sources=[hfthf + "/*.pyx"])
positron_ext = Extension("*", sources=[hphf + "/*.pyx"])

extensions = [decay_ext, gamma_ext, phase_ext,
              field_theory_ext, positron_ext]

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
      include_dirs=[np.get_include(),
                    'hazma/decay_helper_functions',
                    'hazma/positron_helper_functions'],
      include_package_data=True,
      license='MIT License',
      platforms='MacOS and Linux',
      download_url='https://github.com/LoganAMorrison/Hazma',
      classifiers=[
          "Programming Language :: Python",
          "License :: MIT License",
          "Topic :: High Energy Particle Physics"]
      )
