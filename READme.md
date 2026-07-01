# Customer Churn Prediction

Predicting which telecom customers are likely to cancel their subscription, using the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Why this is harder than it sounds

About 73% of customers stay and only 27% churn. That means a model that just guesses "stays" every time would already be 73% accurate — and completely useless. So accuracy isn't the metric that matters here; catching the people who are actually going to leave is.

## What I did

- Cleaned the data (a handful of rows had blank `TotalCharges`, dropped those)
- One-hot encoded the categorical columns
- Split into train/test **before** scaling anything, so no info from the test set leaks into training
- Used a Random Forest to pick out the most useful features and dropped the rest (30 → 15 features)
- Trained a Logistic Regression model with `class_weight='balanced'` so it doesn't just ignore the minority class
- Checked the score held up with 5-fold cross-validation
- Tried a few different `C` values to see if the default regularization was actually the best choice
- Compared against Random Forest as a standalone classifier, not just a feature selector
- Looked at a few different decision thresholds to see the precision/recall tradeoff
- Saved the final model with `joblib`

## Results

| Metric | Score |
|---|---|
| ROC AUC (test set) | 0.83 |
| ROC AUC (5-fold CV avg) | 0.84 |
| Recall on churners | 0.80 |
| Precision on churners | 0.48 |

Logistic Regression ended up beating Random Forest on AUC (0.83 vs 0.81), and it's a lot easier to explain *why* it flagged someone, so that's the final model.

Recall matters more than precision here — missing a customer who's about to churn is more costly than flagging someone who was going to stay anyway. That's why `class_weight='balanced'` is used instead of default settings.
