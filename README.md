🧠 Mental Health Text Classification

Classifying real user posts into mental health categories using NLP and Machine Learning

📌 Overview

This project was developed as part of the AI Dragon Path – Mental Health Text Classification Challenge, where the goal is to automatically classify user-generated posts from mental health support forums into one of five emotional/clinical categories.

The texts contain real discussions of personal emotional struggles, making this a challenging real-world NLP problem due to:

noise (slang, typos, informal language)

long text inputs

overlapping themes

imbalanced classes

This project demonstrates an end-to-end machine learning pipeline from text preprocessing to model training, optimization, evaluation, and final predictions.

📄 Dataset Description

The dataset consists of anonymized posts taken from online mental health communities, including a short title and a longer content field.

Columns
Column	Description
id	Unique identifier for each post
title	Short headline summarizing the post
content	Full text where users describe their emotions and experiences
target	Final label assigned to each training sample
Target Labels (5 Classes)

relationship-and-family-issues ❤️‍🩹
Difficult or toxic family/romantic relationships, emotional conflicts.

depression 🖤
Sadness, hopelessness, fatigue, low self-worth.

anxiety 😰
Overthinking, panic attacks, social anxiety, constant fear.

ptsd-and-trauma ⚠️
Past trauma, violence, flashbacks, emotional scars.

suicidal-thoughts-and-self-harm 🚨
Crisis moments, self-harm ideation, suicidal intent or impulses.

⚠️ Although posts may cover multiple themes, each is labeled with a single best-fit class.

🏁 Competition Goal

Build a model that predicts the correct mental health label for each post in test.csv and submit a CSV file in the format:

id,target
1001,depression
1002,anxiety
...


Evaluation metric: Accuracy
The leaderboard was computed using ~20% of the hidden test set.

🔧 Project Workflow
1️⃣ Data Preprocessing

Cleaning HTML, punctuation, emojis

Normalizing text (lowercasing, contractions, spacing)

Removing stopwords

Lemmatization

Handling missing or corrupted entries

2️⃣ Exploratory Data Analysis (EDA)

Word count distributions

Most common words per category

Class imbalance visualization

Sentiment trends

3️⃣ Handling Class Imbalance

Class weighting

SMOTE / oversampling (when applicable)

Stratified splitting

4️⃣ Vectorization & Representations

Multiple text embeddings tested:

TF-IDF

Word2Vec

DistilBERT / BERT embeddings

SentenceTransformers

5️⃣ Model Training

Tested and compared:

Logistic Regression

SVM

Random Forest

LSTM / BiLSTM

Transformer-based classifiers (BERT, DistilBERT)

Final chosen model:
Logistic Regression

6️⃣ Evaluation

Metrics used:

Accuracy (main metric)

F1-score (macro & weighted)

Confusion matrix

Classification report

7️⃣ Submission Pipeline

One-click script/notebook:

Load test.csv

Preprocess

Predict

Export submission.csv

📁 Repository Structure
mental-health-text-classification/

│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── notebooks/
│   └── dragon-regression (2).ipynb
│
│
├── submission/
│   └── submission.csv
│
└── README.md




🚀 How to Run This Project
Install dependencies
pip install -r requirements.txt

Training
python src/train.py

Prediction
python  --input data/test.csv --output submission.csv

📊 Results

<img width="1372" height="525" alt="image" src="https://github.com/user-attachments/assets/66ebd055-a437-4eea-ad8b-d93aa38c1f4f" />

🌟 Key Learnings

✔ Handling noisy, real-world text
✔ Managing class imbalance in NLP
✔ Using transformer models for long texts
✔ Building a full ML pipeline from dataset → submission
✔ Understanding mental-health specific language patterns

🛡 Ethical Considerations

Because this project uses sensitive mental health data:

No attempts were made to re-identify users

The dataset is anonymized

The model is not intended for clinical use

Goal: educational + research showcase only


