from setuptools import find_packages, setup

setup(
    name="soccer_tasks_mjlab",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["mjlab"],
    python_requires=">=3.10",
)
