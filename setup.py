import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="airball",
    version="0.3.0",
    author="Garett Brown",
    author_email="garett.brown@mail.utoronto.ca",
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
    install_requires=["numpy", "scipy", "rebound", "astropy", "joblib"]
)