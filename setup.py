from setuptools import setup, find_packages

setup(
    name="dummy_collectives",
    version="0.0.1",
    description="A dummy PyTorch distributed backend implementation",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    py_modules=["dummy_backend"],
    install_requires=[
        "torch>=1.8.0",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
