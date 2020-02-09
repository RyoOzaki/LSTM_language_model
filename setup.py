from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='LSTM_language_model',
    version='2.0.0',
    description='Package of the LSTM_language_model',
    long_description=readme,
    author='Ryo Ozaki',
    author_email='ryo.ozaki@em.ci.ritsumei.ac.jp',
    url='https://github.com/RyoOzaki/LSTM_language_model',
    license=license,
    install_requires=['numpy', 'tensorflow>=2.1.0'],
    packages=['LSTM_language_model',]
)
