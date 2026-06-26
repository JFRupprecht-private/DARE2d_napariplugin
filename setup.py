#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="dare2d",
    version="1.0",
    description="DARE2D: Division Axis and Region Estimation in 2D time-lapse images (core)",
    author="Romain Karpinski, Alice Gros, Marc Karnat, Qazi Saaheelur Rahaman, "
    "Jules Vanaret, Mehdi Saadaoui, Sham Tlili, Jean-Francois Rupprecht",
    url="https://github.com/qazi05/DARE2d",
    packages=find_packages(include=["dare2d", "dare2d.*"]),
)
