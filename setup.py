from setuptools import setup, find_packages

setup(
    name="istrust_model",
    version="1.0",
    description="Deep monotonic unsupervised clustering model for feature extraction and clustering analysis of deteriorating systems.",
    authors="Panagiotis Komninos, A.E.C. Verraest"
    author_emails="panakom@hotmail.com, A.E.C.Verraest@student.tudelft.nl"
    packages=find_packages(include=["istrust_model", "istrust_model.*"]),
    install_requires=[
        "pandas==1.5.3",
        "matplotlib==3.5.2",
        "pillow==9.4.0",
        "scipy==1.10.1",
        "keyboard==0.13.5",
        "umap-learn==0.5.3",
        "joblib==1.2.0",
        "numpy==1.23.4",
        "tqdm==4.64.0",
    ],
)
