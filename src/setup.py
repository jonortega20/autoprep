"""Setup script for the data processing library."""

from setuptools import setup, find_packages

setup(
    name="data_processing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "sphinx_rtd_theme>=3.0.1",
    ],
    author=["Jon Ortega", "Haritz Bolumburu", "Iker Barrero", "Jon Tobalina"],
    author_email="jonortega20@gmail.com",
    description="A library for automated data preprocessing and initial analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/stackblitz/data-processing",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)