from setuptools import setup, find_packages

setup(
    name="cptliq",
    version="0.1.0",
    description="CPT-based liquefaction evaluation (Robertson & Wride 1998 / Youd et al. 2001)",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=["numpy>=1.24"],
)
