from setuptools import setup, find_packages

setup(
    name="istrust_model",
    version="1.0",
    description="A novel interpretable transformer-based model for Remaining Useful Life (RUL) prediction from raw sequential images (frames) representing a composite structure under fatigue loads",
    author="Panagiotis Komninos, A.E.C. Verraest"
    author_email="panakom@hotmail.com"
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
