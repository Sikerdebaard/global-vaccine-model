from setuptools import find_packages, setup
import sys

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported')

#with open("README.md", "r") as fh:
#    long_description = fh.read()


setup(
    name='covid19-vaccination-model',
    version='1.0.0',
    description='Estimates the vaccination rollout for the COVID-19 vaccines',
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    author='Thomas Phil',
    author_email='thomas@tphil.nl',
    url='https://github.com/Sikerdebaard/dcmrtstruct2nii',
    python_requires=">=3.6",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        'matplotlib>=3.3.4',
        'numpy>=1.20.0',
        'pandas>=1.2.1',
    ],
    entry_points={
        'console_scripts': [
            'covid-19-model-single-country=cli.model_country:cli',
            'covid-19-add-missing-countries=cli.add_missing_countries:cli',
        ],
    },
)
