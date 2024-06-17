"""setup.py file for the ``vocab_augmentor`` package.
"""
import fnmatch
import os
import sys
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as build_py_orig

from vocab_augmentor import __version__

MAJOR = 3
MINOR = 10
PATCH = 13

VERSION = __version__
excluded = []

if sys.version_info < (MAJOR, MINOR, PATCH):
    raise RuntimeError(f"vocab_augmentor v{VERSION}+ supports Python {MAJOR}.{MINOR}.{PATCH} and above.")


# IMPORTANT: bdist_wheel behaves differently to sdist
# - MANIFEST.in works for source distributions, but it's ignored for wheels,
#   See https://bit.ly/3s2Kt3p
# - "bdist_wheel would always include stuff that I'd manage to exclude from the sdist"
#   See https://bit.ly/3t7SsO4
# Code reference: https://stackoverflow.com/a/56044136/14664104
class build_py(build_py_orig):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, mod, file)
            for (pkg, mod, file) in modules
            if not any(fnmatch.fnmatchcase(file, pat=pattern) for pattern in excluded)
        ]


# Directory of this file
dirpath = os.path.abspath(os.path.dirname(__file__))

# The text of the README file
with open(os.path.join(dirpath, "README.md"), encoding="utf-8") as f:
    README = f.read()

# The text of the requirements.txt file
with open(os.path.join(dirpath, "requirements.txt")) as f:
    REQUIREMENTS = f.read().splitlines()

setup(name='vocab-augmentor',
      version=VERSION,
      description='''Translates and expands your vocabulary list by identifying and 
      translating new words from provided text using various language models.''',
      long_description=README,
      long_description_content_type='text/markdown',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
      ],
      keywords='language vocabulary translation segmentation language-learning natural-language-processing text-processing education',
      url='https://github.com/raul23/vocab-augmentor',
      author='Raul C.',
      author_email='rchfe23@gmail.com',
      license='MIT',
      python_requires='>=3.10.13',
      packages=find_packages(exclude=['tests']),
      cmdclass={'build_py': build_py},
      include_package_data=True,
      exclude_package_data={'': ['docs/*']},
      install_requires=REQUIREMENTS,
      entry_points={
        'console_scripts': ['vocab=vocab_augmentor.scripts.vocab:main']
      },
      project_urls={  # Optional
          'Bug Reports': 'https://github.com/raul23/vocab-augmentor/issues',
          'Source': 'https://github.com/raul23/vocab-augmentor',
      },
      zip_safe=False)
