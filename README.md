
# Logistics Customer Issue Classifier (NLP Task)

## Problem
Automatically classify customer feedback into issue types (e.g. Delivery Delay, Package Condition Issue) for better routing in logistics support.

## Dataset Used
- Customer Sentiment Dataset (Kaggle)
- Link: https://www.kaggle.com/datasets/kundanbedmutha/customer-sentiment-dataset
- 25,000 rows, but **only 15 unique review texts** repeated many times (avg length ~32 chars).
- Key column: `review_text`

## Approach
1. EDA → discovered extreme repetition (15 templates)
2. Rule-based labeling → 5 categories + Other/General using keywords
3. Text cleaning (lowercase, remove stopwords/punctuation)
4. TF-IDF vectorization (bigrams)
5. Logistic Regression with balanced class weights

## Results
- Accuracy: 100.00%
- Macro F1-score: 1.0000
- Weighted F1-score: 1.0000
- All classes perfectly classified on test set

## Limitations
- Dataset is synthetic: only **15 unique short phrases** repeated thousands of times → model memorizes patterns, not learns general language.
- Very short reviews (avg 32 chars) → limited context for NLP.
- Perfect score is due to data repetition, not real-world robustness.
- In actual logistics feedback, expect lower performance (70–85% F1) due to diversity, typos, sarcasm.

## What I would improve with more time
- Use real, diverse review data (e.g. Amazon/Flipkart delivery reviews)
- Text augmentation to create variation
- Fine-tune DistilBERT or similar for better short-text handling
- Add multilabel support if reviews have multiple issues
- Deploy simple inference API

## Repository Structure
- `issue_classifier.ipynb` — full notebook
- Dataset: Download from Kaggle link above
