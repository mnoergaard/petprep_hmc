from setuptools import setup, find_packages

setup(
    name="petprep_hmc",
    version="0.0.10",
    description="PETPrep Head Motion Correction Workflow",
    author="Martin Norgaard",
    author_email="martin.noergaard@di.ku.dk",
    url="https://github.com/mnoergaard/petprep_hmc",
    packages=find_packages(),
    install_requires=[
        "click==8.1.3",
        "contourpy==1.0.7",
        "matplotlib==3.7.1",
        "networkx==3.1",
        "nibabel==5.1.0",
        "nipype==1.8.6",
        "numpy==1.24.2",
        "pandas==2.0.0",
        "patsy==0.5.3",
        "pillow==9.5.0",
        "scipy==1.10.1",
        "seaborn==0.12.2",
        "statsmodels==0.13.5",
        "traits==6.3.2",
        "rdflib==6.3.2",
        "astor==0.8.1",
        "bids==0.0",
        "bids-validator==1.11.0",
        "docopt==0.6.2",
        "formulaic==0.5.2",
        "interface-meta==1.3.0",
        "num2words==0.5.12",
        "pybids==0.15.6",
        "sqlalchemy==1.3.24",
        "wrapt==1.15.0",
        "niworkflows==1.11.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            "sphinx",
            "myst-parser",
            "black",
            # Add any other development/testing packages here
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
