from setuptools import setup, find_packages

setup(
    name="classify5",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.65.0",
        "pytest>=6.2.0",
        "scipy>=1.7.0",
        "PyYAML>=6.0",
        "textual==0.47.1",
        "rich>=13.6.0",
        "scikit-learn>=1.0.0",
    ],
)
