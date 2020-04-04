import os
from setuptools import setup, find_packages


DESCRIPTION = "Utilities for developing an MLsploit python module."


def read_version():
    with open("mlsploit/__init__.py") as f:
        for line in f.readlines():
            if line.startswith("__version__"):
                version = line.split("=")[1].strip(" \"'\n")
                return version
    raise RuntimeError("Cannot read version string.")


def load_readme():
    with open("README.md") as f:
        readme = f.read()

    return readme


def load_requirements():
    with open("requirements.txt") as f:
        requirements = f.readlines()

    requirements = map(lambda r: r.strip(), requirements)
    requirements = filter(len, requirements)

    return list(requirements)


setup(
    name="mlsploit-py",
    version=read_version(),
    license="BSD-3-Clause",
    author="Nilaksh Das",
    url="https://github.com/mlsploit/mlsploit-py",
    python_requires=">=3.6, <4",
    install_requires=load_requirements(),
    description=DESCRIPTION,
    long_description=load_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=("mlsploit", "mlsploit.*")),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: BSD License",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
