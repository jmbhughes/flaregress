#!/usr/bin/env python

from distutils.core import setup

setup(
    name='flaregress',
    version='0.0.1',
    author='J. Marcus Hughes',
    author_email='hughes.jmb@gmail.com',
    packages=['flaregress'],
    url='',
    license='LICENSE.txt',
    description='Software to load solar flare data and predict their future lightcurves',
    long_description=open('README.md').read(),
    install_requires=["numpy",
                      "pandas",
                      "sunpy",
                      "scikit-learn",
                      "deepdish",
                      "statsmodels"],
    test_suite="tests"
)