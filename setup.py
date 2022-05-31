from setuptools import setup, find_packages


try:
    with open('README.md') as f:
        readme = f.read()
except IOError:
    readme = ''


name = 'larch'
exec(open('larch/_version.py').read())
release = __version__
version = '.'.join(release.split('.')[:2])

with open('requirements.txt') as fin:
    requirements = [line.strip() for line in fin]

setup(
    name=name,
    author='Yuta Koreeda',
    author_email='yuta.koreeda.pb@hitachi.com',
    maintainer='Yuta Koreeda',
    maintainer_email='yuta.koreeda.pb@hitachi.com',
    version=release,
    description='Automatic readme generation using language models',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/hitachi-nlp/larch',
    packages=find_packages(),
    install_requires=requirements,
    entry_points = {
        'console_scripts': [
            'larch=larch.cli:cli',
            'larch-server=larch.server:cli',
            'larch-server-dryrun=larch.server:init_dryrun'
        ],
    },
    classifiers=[
        "Environment :: Console",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ]
)
