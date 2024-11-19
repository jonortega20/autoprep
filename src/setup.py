"""Setup script for the data processing library."""

from distutils.core import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="autoprep",
    packages=["autoprep"],
    version="0.2.0",
    license= 'GNU',
    description="A library for automated data preprocessing and initial analysis",
    author=["Jon Ortega", "Haritz Bolumburu", "Iker Barrero", "Jon Tobalina"],
    author_email="jonortega20@gmail.com",
    url="https://github.com/jonortega20/autoprep",
    #download_url="
    keywords=["data preprocessing", "data analysis", "data cleaning", "data wrangling", "outliers", "missings"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.2.3",
        "matplotlib>=3.9.2",
        "seaborn>=0.13.2",
        "scikit-learn>=1.5.2",
        "scipy>=1.14.1",
        "statsmodels>=0.14.4",
        "sphinx_rtd_theme>=3.0.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: GNU Affero General Public License v3",
    ],
    python_requires=">=3.7",
)