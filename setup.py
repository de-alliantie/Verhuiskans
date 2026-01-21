from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="src",
    description="Voorspellen van mutaties",
    long_description=long_description,
    author="Elham Wasei; Thomas Westveer; Jeroen Vranken",
    packages=find_packages(),
    version="0.1.0",
    license="Copyright (C) 2025 De Alliantie",
    python_requires=">=3.10",
)
