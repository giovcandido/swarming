from setuptools import setup, find_packages

from os.path import join

# Load version
exec(open(join('swarming', 'version.py')).read())

# Load long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Load requirements
with open('requirements.txt', encoding='utf-8') as f:
    requirements = f.read()

# Configure setup
setup(
    name = 'swarming',
    version = __version__, # type: ignore
    author = 'Giovani Candido',
    author_email = 'giovcandido@outlook.com',
    license = 'GNU General Public License v3.0',
    description = (
        'Swarming is a library that features both parallel '
        'and serial implementation of the PSO algorithm.'
    ),
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/giovcandido/swarming',
    packages = find_packages(),
    install_requires = [requirements],
    python_requires = '>=3.8',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.8',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ]
)
