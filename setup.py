import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vibrato-space", # Replace with your own username
    version="0.0.1",
    author="Max Henry",
    author_email="max.henry@mail.mcgill.ca",
    description="Tools for building a vibrato perceptual space experiment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxsolomonhenry/vibrato_space",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
