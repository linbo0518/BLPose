import os
import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

VERSION = "20210405"
README = open("README.md").read()
REQUIREMENTS = [
    "albumentations",
    "cython",
    "numpy",
    "opencv-python",
    "pycocotools",
    "torch",
]

package_dir = os.path.join(os.path.dirname(__file__), "blpose")
submodule_dir = os.path.join(package_dir, "libpose")
source_file = os.path.join(submodule_dir, "libpaf_cpu.pyx")

ext = [
    Extension(
        name="blpose.libpose",
        sources=[
            source_file,
        ],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="blpose",
    version=VERSION,
    author="Leon Lin",
    author_email="linbo0518@gmail.com",
    url="https://github.com/linbo0518/BLPose",
    description="linbo0518@gmail.com",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    ext_modules=cythonize(ext),
    install_requires=REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)