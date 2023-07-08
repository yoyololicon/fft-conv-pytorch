from os import name
import setuptools

NAME = "torch_fftconv"
VERSION = '0.1.3'
MAINTAINER = 'Chin-Yun Yu'
EMAIL = 'lolimaster.cs03@nctu.edu.tw'


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=MAINTAINER,
    author_email=EMAIL,
    description="Implementation of 1D, 2D, and 3D FFT convolutions in PyTorch. Much faster than direct convolutions for large kernel sizes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyololicon/fft-conv-pytorch",
    packages=["torch_fftconv"],
    install_requires=['torch>=1.7.0'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
