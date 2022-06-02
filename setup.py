import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="airball",
    version="0.0.3",
    author="Garett Brown",
    author_email="gbrown@physics.utoronto.ca",
    description="A package for implementing flybys in hannorein/rebound",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zyrxvo/airball/",
    project_urls={
        "Bug Tracker": "https://github.com/zyrxvo/airball/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "rebound>=3.18",
        "numpy>=1.22.4",
        "scipy>=1.8.1",
        "matplotlib>=3.5.2"
    ]
)