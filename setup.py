from setuptools import setup, find_packages

setup(
    name="ElegansBot",
    version="1.0.2",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'numba>=0.54.0',
        'scipy>=1.5.0',
        'matplotlib>=3.4.2',
    ],
    author="Taegon Chung",
    author_email="sunny2ys@dgist.ac.kr",
    description="Newtonian Mechanics Model for C. elegans Locomotion",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TaegonChung/ElegansBot",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)