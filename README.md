# Loan Default Analysis Project

A data science project analyzing consumer loan data to identify factors contributing to loan defaults and building predictive models using CatBoost.

## Overview

This repository contains a comprehensive analysis of loan data with the goal of understanding default risk factors. The project includes exploratory data analysis, statistical testing, and machine learning models to predict loan defaults.

## Project Structure

```
├── data/                  # Dataset directory
│   └── loans_fs.csv       # Loan dataset
├── notebooks/             # Jupyter notebooks
│   ├── analysis.ipynb     # EDA and statistical analysis
│   └── catboost.ipynb     # Machine learning models with CatBoost
├── util/                  # Utility functions
├── pyproject.toml         # Project dependencies
└── README.md              # Project documentation
```

## Installation

This project uses Python 3.12 and [uv](https://github.com/astral-sh/uv) for dependency management.

### Setup with uv

1. Install uv if you don't have it already:
```bash
curl -sSf https://astral.sh/uv/install.sh | bash
```

2. Clone the repository:
```bash
git clone <repository-url>
cd loan-analysis
```

3. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## Features

- **Exploratory Data Analysis**: Comprehensive analysis of loan characteristics and their relationship to defaults
- **Statistical Testing**: Chi-square tests for categorical variables and t-tests for numerical variables
- **Machine Learning**: CatBoost classifier for loan default prediction with high-cardinality categorical feature handling
- **Visualizations**: Distribution plots, correlation analysis, and feature importance charts

## Dataset

The analysis uses a loan dataset (`loans_fs.csv`) containing information about:

- Loan amount
- Interest rates
- Debt-to-income ratio (DTI)
- Annual income
- Home ownership status
- Loan term
- Loan purpose
- Employment information
- Default status

## Usage

1. Start by exploring the data analysis notebook:
```bash
jupyter notebook notebooks/analysis.ipynb
```

2. To run the machine learning model:
```bash
jupyter notebook notebooks/catboost.ipynb
```

## Model Details

The CatBoost model includes:
- Preprocessing for numerical and categorical features
- Power transformation for skewed features
- Handling of high-cardinality categorical variables
- Feature importance analysis
- ROC curve evaluation

## Dependencies

Main dependencies include:
- polars
- catboost
- scikit-learn
- matplotlib
- seaborn
- shap
- numpy
- scipy

See [`pyproject.toml`](pyproject.toml) for the complete list.
