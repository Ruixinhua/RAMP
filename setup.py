import setuptools

with open("README.md", "r", encoding="utf-8") as fd:
    long_description = fd.read()

setuptools.setup(
    name="ramp-fuxictr",
    version="2.3.9",
    author="Ruixinhua",
    author_email="Ruixinhua@users.noreply.github.com",
    description="RAMP code release built on FuxiCTR for robust ad recommendation under limited personalized-feature availability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ruixinhua/RAMP",
    download_url="https://github.com/Ruixinhua/RAMP/tags",
    packages=setuptools.find_packages(
        exclude=["model_zoo", "tests", "data", "docs", "demo"]),
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=["keras_preprocessing", "pandas", "PyYAML>=5.1", "scikit-learn",
                      "torchmetrics", "numpy", "h5py", "tqdm", "pyarrow", "polars"],
    classifiers=(
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ),
    license="Apache-2.0 License",
    keywords=['ctr prediction', 'recommender systems',
              'ctr', 'cvr', 'pytorch'],
)
