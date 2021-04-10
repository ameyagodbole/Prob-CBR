# !/usr/bin/env python

from distutils.core import setup

setup(name='prob_cbr',
      version='0.02',
      packages=['prob_cbr'],
      install_requires=[
          "wandb",
          "numpy",
          "scipy",
          "tqdm"
      ],
      package_dir={'prob_cbr': 'prob_cbr'}
      )
