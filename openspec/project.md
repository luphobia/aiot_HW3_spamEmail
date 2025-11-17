# Project Context

## Purpose
Build a machine learning-based Spam Email Classifier to automatically detect and filter spam emails, improving user productivity and security.

## Tech Stack
 - Python 3.x
 - scikit-learn (ML library)
 - pandas (data handling)
 - pytest (testing)
 - Jupyter Notebook (optional for prototyping)

## Project Conventions

### Code Style
 - Follow PEP8 for Python code
 - Use descriptive variable and function names
 - Include docstrings for all public functions/classes

### Architecture Patterns
 - Start with a single-file implementation
 - Modularize into data loading, feature engineering, model training, and prediction as needed

### Testing Strategy
 - Unit tests for data processing and model logic (pytest)
 - Use sample datasets for test coverage

### Git Workflow
 - Feature branches for new capabilities (e.g., `feature/spam-classifier`)
 - Descriptive commit messages (imperative mood)
 - Pull requests for code review

## Domain Context
 - Email data may contain sensitive information; ensure privacy and security
 - Classifier should handle common spam patterns and adapt to new ones

## Important Constraints
 - Must comply with data privacy regulations (e.g., GDPR)
 - Avoid storing raw email data longer than necessary

## External Dependencies
 - scikit-learn, pandas (Python packages)
 - Email datasets (e.g., Enron, SpamAssassin)
