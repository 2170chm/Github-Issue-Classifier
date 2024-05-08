# GitHub-Issue-Classifier

## Overview
The GitHub-Issue-Classifier is a machine learning tool designed to automatically classify issues on GitHub based on their content. It utilizes advanced natural language processing (NLP) techniques to preprocess the text of issues, and several machine learning models to determine the most appropriate labels for each issue. This classifier helps manage and triage incoming GitHub issues more efficiently.

## Features
- **Text Processing Techniques**: Cleans and prepares text data by removing punctuations, stopwords, and applying lemmatization to reduce words to their base forms.
- **Text Representations**:
  - TF-IDF Vectors (Bag of Words): Transforms text into a meaningful representation of numbers which is easy to compare.
  - Topic Models (LDA): Uses Latent Dirichlet Allocation to discover abstract topics within text.
  - Word Embeddings: Utilizes dense representations of words that capture contextual nuances.
- **Machine Learning Models**:
  - Random Forests
  - Gaussian Naive Bayes
  - Neural Network
  - Support Vector Machine (SVM)

The system trains on a dataset using combinations of these models and settings to identify the best performing model and setting configuration.

## Prerequisites
Ensure you have Python installed on your system to run the scripts. You can download Python [here](https://www.python.org/downloads/).

## Getting Started

### Step 1: Clone the Repository
To get started, clone this repository to your local machine using the following command:
```
git clone [repository-url]
cd GitHub-Issue-Classifier
```
Replace `[repository-url]` with the actual URL of the repository.

### Step 2: Install Dependencies
Install the necessary Python libraries using:
```
pip install -r requirements.txt
```

### Step 3: Run the Classifier
Execute the script by running:

```
python github_issue_classifier.py
```
Follow the on-screen instructions to input issue data and receive classifications.

## How It Works
1. **Data Preprocessing**: The script first preprocesses the text data using the specified text processing techniques.
2. **Feature Extraction**: It then converts the preprocessed text into one of the selected representations.
3. **Model Training**: Various models are trained using combinations of preprocessing techniques and text representations.
4. **Evaluation and Selection**: The models are evaluated, and the best performing combination is selected for use.
5. **Classification**: New issues are classified using the selected model and settings.
