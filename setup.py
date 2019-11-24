import os
import re
import setuptools
import subprocess

# Directory containing data files for this package.
DATA_DIR = "pyunits/data"
# Path to the VERSION file, which will be automatically added by this
# script when building a wheel or sdist, and automatically read when installing
# to determine the version.
VERSION_FILE = os.path.abspath(os.path.join(DATA_DIR, "VERSION"))

# Regex that matches valid PEP-440 versions.
# Taken from here:
# https://www.python.org/dev/peps/pep-0440/#appendix-b-parsing-version-strings-with-regular-expressions
VERSION_RE = re.compile(
    r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)"
    r"(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$")


def get_version() -> str:
    """
    Read the current tagged version from Git, if available, or from the version
    file, if not.
    :return: The PEP-440 version string.
    """
    try:
        version = subprocess.check_output(["/usr/bin/git", "describe", "--tags"]
                                          )
        # Remove trailing newline.
        version = version.decode("utf8").rstrip("\n")

        # Create the version file.
        with open(VERSION_FILE, "w") as version_file:
            version_file.write(version)

    except (subprocess.CalledProcessError, FileNotFoundError):
        # We are not in a Git repo, or Git is not installed. Use the VERSION
        # file.
        with open(VERSION_FILE, "r") as version_file:
            version = version_file.read()

    # Check that the version is valid.
    if VERSION_RE.match(version) is None:
        raise ValueError("Version '{}' is invalid. Please tag a valid version."
                         .format(version))

    print("Tagging with version: {}".format(version))
    return version


def get_long_description() -> str:
    """
    Gets the full description to use for this package, which is just the
    contents of the README file.
    :return: The full description.
    """
    with open("README.md", "r") as fh:
        return fh.read()


setuptools.setup(
    name="python-pyunits",
    version=get_version(),
    author="Daniel Petti",
    author_email="djpetti@gmail.com",
    description="A package for making Python unit-aware.",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/djpetti/pyunits",
    packages=setuptools.find_packages(exclude=["*.tests", "examples"]),
    package_data={"pyunits": ["data/*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=["numpy", "loguru", "pytest", "pytest-cov"]
)
