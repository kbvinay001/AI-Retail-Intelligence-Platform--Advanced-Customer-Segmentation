from setuptools import setup, find_packages

setup(
    name="ai-retail-intelligence",
    version="3.0.0",
    author="kbvinay001",
    author_email="kbhaskarvinay@gmail.com",
    description="Enterprise AI-powered retail analytics platform",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kbvinay001/AI-Retail-Intelligence-Platform--Advanced-Customer-Segmentation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.1.0",
        "plotly>=5.10.0",
        "scipy>=1.9.0",
    ],
    extras_require={
        "api": ["fastapi>=0.95.0", "uvicorn[standard]>=0.21.0", "python-jose[cryptography]>=3.3.0",
                "passlib[bcrypt]>=1.7.4", "cryptography>=40.0.0"],
        "forecasting": ["prophet>=1.1.1", "pmdarima>=2.0.0", "xgboost>=1.7.0"],
        "full": ["fastapi>=0.95.0", "uvicorn[standard]>=0.21.0", "prophet>=1.1.1",
                 "openai>=1.0.0", "cryptography>=40.0.0", "python-jose[cryptography]>=3.3.0",
                 "passlib[bcrypt]>=1.7.4", "python-dotenv>=1.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
