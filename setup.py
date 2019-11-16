import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="python-pyunits",
    version="0.1rc2",
    author="Daniel Petti",
    author_email="djpetti@gmail.com",
    description="A package for making Python unit-aware.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/djpetti/pyunits",
    packages=setuptools.find_packages(exclude=["*.tests", "examples"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=["numpy", "loguru", "pytest", "pytest-cov"]
)
