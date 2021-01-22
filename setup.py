import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vibratospace", # Replace with your own username
    version="0.0.1",
    author="MaxHenry",
    author_email="max.henry@mail.mcgill.ca",
    description="Tools for pitch trajectory transfer",
    long_description=long_description,
    keywords="core package",
    long_description_content_type="text/markdown",
    url="https://github.com/maxsolomonhenry/vibrato_space",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'scipy==1.4.1',
        'numpy==1.18.1',
        'tqdm==4.46.0',
        'crepe==0.0.11',
        'matplotlib==3.2.1',
        'librosa==0.7.2'
    ],
    python_requires='>=3.7',
)