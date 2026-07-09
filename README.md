# Twitter Sentiment Analysis Project

A machine learning project that performs sentiment analysis on Twitter data. This repository follows MLOps best practices by using Git for version controlling source code and **DVC (Data Version Control)** for tracking datasets and model artifacts.

## 🚀 Project Overview
This project processes Twitter text data to classify the underlying sentiment (happiness & sadness). 
*   **Model:** XGBoost

## ⚙️ Development Workflow & Architecture

This project follows a structured MLOps workflow, moving from initial experimentation to an automated production pipeline:

1. **Prototyping & Experimentation:** Initial data analysis, text cleaning exploration, feature engineering,  and model evaluation were conducted using Jupyter Notebook to solve the core data science problem. By prototyping in a Jupyter Notebook first, the hardest data science problems were solved: figuring out how to clean the text, which features matter.
2. **Modularization:** The experimental notebook code was refactored into modular, production-grade Python scripts (‘data_ingestion.py’,  ‘data_preprocessing.py’, ‘feature_engineering.py’,  ‘model_building.py’, ‘model_evaluation.py). This engineering work makes the project automated, maintainable, reproducible, and ready for DVC.
3. **Data Version Control (DVC):** Datasets and model artifacts are decoupled from Git and version-controlled via DVC pipelines to ensure tracking reproducibility across environments.

## 🛠️ Tech Stack
*   **Language:** Python
*   **ML Libraries:** [e.g., Scikit-learn, NLTK, Pandas]
*   **Version Control:** Git & GitHub (Code)
*   **Data Versioning:** DVC (Data and Model weights)

## 📦 Data & Model Pipeline (DVC)
Tthe datasets and trained model files are tracked via DVC. 

This keeps the codebase lightweight and reproducible.

# Build & Track ML Pipelines with DVC

## How to run?

conda create -n test python=3.11 -y

conda activate test

## pip install -r requirements.txt
python -m pip install -r requirements.txt

## DVC Commands

git init

dvc init

dvc repro

dvc dag

dvc metrics show
