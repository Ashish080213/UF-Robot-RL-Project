# setup.py

from setuptools import setup, find_packages

setup(
    name="lite6_gym",
    version="0.1",
    packages=find_packages(),  # This picks up envs/, tests/, etc.
    install_requires=[
        "gymnasium",  # or gym if not using gymnasium
        "pybullet",
        # ...other dependencies
    ],
)
