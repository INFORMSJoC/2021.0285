#  ___________________________________________________________________________
#
#  Parapint
#  Copyright (c) 2020
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from setuptools import setup, find_packages


setup(name='parapint',
      version='0.4.0',
      packages=find_packages(),
      description='Parapint: Parallel NLP algorithms',
      author='Parapint Developers',
      maintainer_email='mlbynum@sandia.gov',
      license='Revised BSD',
      url='https://github.com/parapint/parapint',
      install_requires=['numpy>=1.13.0', 'scipy', 'pyomo>=6.4', 'mpi4py'],
      include_package_data=True,
      python_requires='>=3.7',
      classifiers=["Programming Language :: Python :: 3",
                   "Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: 3.8",
                   "Programming Language :: Python :: 3.9",
                   "License :: OSI Approved :: BSD License",
                   "Operating System :: OS Independent"]
      )
