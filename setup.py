from distutils.core import setup

requirements = [
    "numpy",
    "matplotlib"
]

setup(
    author="Josef Djärf",
    author_email="josef.djarfs@gmail.com",
    name="semg_sim",
    version="0.0.1",
    packages=["semg_sim"],
    license="MIT",
    long_description=open("README.md").read(),
    install_requires=requirements
)