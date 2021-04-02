#!/usr/bin/env python
"""The setup script."""

from setuptools import setup, find_packages


with open('README.rst') as readme_file:
    long_description = readme_file.read()

requirements = [
    'numpy>=1.9',
    'scipy'  #TODO get minimum required scipy version
]

setup_requirements = [
]

test_requirements = [
    'pytest',  #TODO get minimum required pytest version
]

setup(
    author='Donald Erb',
    author_email='donnie.erb@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    description='A collection of algorithms for fitting the baseline of experimental data.',
    install_requires=requirements,
    license='BSD 3-Clause',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords=[
        'materials characterization',
        'baseline',
        'background',
        'baseline subtraction',
        'background subtraction'
    ],
    name='pybaselines',
    packages=find_packages(include=['pybaselines', 'pybaselines.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/derb12/pybaselines',
    version='0.2.0',
    zip_safe=False,
)
