from setuptools import setup, find_packages

setup(
    name="kepin",
    version="1.0.0",
    description="KePIN: Koopman-Enhanced Physics-Informed Network",
    author="Rahul",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.16",
        "keras>=3.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.8",
        "scipy>=1.11",
    ],
    entry_points={
        "console_scripts": [
            "kepin-train=scripts.train:main",
            "kepin-evaluate=scripts.evaluate:main",
            "kepin-ablation=scripts.ablation:main",
        ],
    },
)
