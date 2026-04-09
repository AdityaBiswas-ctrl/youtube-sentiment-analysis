from setuptools import setup, find_packages

setup(
    name="youtube-sentiment-analysis",
    version="1.0.0",
    description="End-to-end YouTube comment sentiment analysis with MLOps pipeline",
    author="YouTube Sentiment Analyzer",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "flask>=3.1",
        "flask-cors>=5.0",
        "google-api-python-client>=2.100",
        "vaderSentiment>=3.3",
        "scikit-learn>=1.5",
        "xgboost>=2.0",
        "lightgbm>=4.0",
        "pandas>=2.0",
        "numpy>=2.0",
        "joblib>=1.3",
        "mlflow>=2.10",
        "dvc>=3.50",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "flake8>=7.0",
        ],
    },
)
