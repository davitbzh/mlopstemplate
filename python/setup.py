import os
import imp
from setuptools import setup, find_packages

__version__ = imp.load_source(
    "homedepot.version", os.path.join("homedepot", "version.py")
).__version__


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="mlopstemplate",
    version=__version__,
    python_requires=">=3.7,<3.11",
    install_requires=[
        "hopsworks",
    ],
    author="Hopsworks AB",
    author_email="davit@hopsworks.ai",
    description="Home Depot",
    license="GNU GENERAL PUBLIC LICENSE Version 3.0",
    keywords="Hopsworks, Feature Store, Spark, Machine Learning, MLOps, DataOps",
    url="https://github.com/davitbzhs/mlopstemplate",
    download_url="https://github.com/davitbzh/mlopstemplate/releases/tag/"
                 + __version__,
    packages=find_packages(exclude=["tests*"]),
    long_description=read("../README.md"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Developers",
    ],
)
