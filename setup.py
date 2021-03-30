import setuptools
from torch_fftconv import __version__, __email__, name, __maintainer__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=name,
    version=__version__,
    author=__maintainer__,
    author_email=__email__,
    description="Implementation of 1D, 2D, and 3D FFT convolutions in PyTorch. Much faster than direct convolutions for large kernel sizes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yoyololicon/fft-conv-pytorch",
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.7.0'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)