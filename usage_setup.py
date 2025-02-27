from setuptools import setup, find_packages

setup(
    name="Dummy-Collectives",
    version="0.0.1",
    py_modules=["dummy_collectives"],
    install_requires=[
        "torch>=2.6.0",
    ],
    python_requires='>=3.10',
)
