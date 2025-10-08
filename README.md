# Fake News Detection using DistilBERT, RoBERTa, and Ensemble Learning

Fake News Detection Using Ensemble Models
Project Overview

This project is designed to detect fake news using an ensemble of two transformer-based models: DistilBERT and RoBERTa. By combining predictions from both models, the ensemble improves accuracy and reliability in classifying news articles as either real or fake.

**Features**

Uses DistilBERT and RoBERTa pre-trained transformer models.

Fine-tuned on a custom dataset of fake and real news.

Implements an ensemble approach by averaging predictions from both models.

Provides accuracy metrics, confusion matrix, and optional visualizations.

Fast training using a reduced dataset subset for experimentation.

**Dataset**

The dataset consists of two CSV files: Fake.csv and True.csv.

Each entry contains a title and text of a news article.

**Labels:**

0 or fake → Fake news

1 or real → Real news

**How It Works**

The dataset is loaded and preprocessed.

Both models (DistilBERT and RoBERTa) are fine-tuned on the training data.

During inference, the ensemble averages the predictions from both models.

The final output classifies news articles as real or fake.

**Results**

The ensemble achieves high accuracy on the evaluation dataset.

Generates a confusion matrix and classification report for performance analysis.
