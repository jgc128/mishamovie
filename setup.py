import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

with open("VERSION.txt", "r") as fh:
    build_version = fh.read().strip()

setuptools.setup(
    name="mishamovie",
    install_requires=requirements,
    version=build_version,
    author="Alexey Romanov",
    author_email="jgc128@outlook.com", 
    description="Face processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://dev.azure.com/cisl/Raven/_git/toucan-exp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)
