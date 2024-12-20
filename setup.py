from setuptools import find_packages, setup


def read_requirements(filename: str):
    with open(filename) as requirements_file:
        requirements = []
        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) <= 0:
                continue
            requirements.append(line)
    return requirements


# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import cached_path whilst setting up.
VERSION = {}  # type: ignore
with open("torchmsat/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

setup(
    name="torchmsat",
    version=VERSION["VERSION"],
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="",
    url="https://github.com/scale-lab/torchmsat",
    author="Abdelrahman Hosny",
    author_email="abdelrahman_hosny@brown.edu",
    license="Apache",
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"],
    ),
    package_data={"torchmsat": ["py.typed"]},
    install_requires=read_requirements("requirements.txt"),
    extras_require={"dev": read_requirements("dev-requirements.txt")},
    python_requires=">=3.8",
)
