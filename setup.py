from setuptools import setup, find_packages

setup(
    name="ISTRUST_MODEL",
    version="1.0",
    description="Deep monotonic unsupervised clustering model for feature extraction and clustering analysis of deteriorating systems.",
    authors="Panagiotis Komninos, A.E.C. Verraest"
    author_emails="panakom@hotmail.com, A.E.C.Verraest@student.tudelft.nl"
    packages=find_packages(include=["monotonic_dc", "monotonic_dc.*"]),
    install_requires=[
        "pandas==1.5.3",
        "matplotlib==3.5.2",
        "seaborn==0.12.1",
        "scikit-learn==1.2.2",
        "scikit-survival==0.21.0",
        "hmmlearn==0.3.0",
        "tslearn==0.5.2",
        "joblib==1.2.0",
        "numpy==1.23.4",
        "tqdm==4.64.0",
    ],
)
