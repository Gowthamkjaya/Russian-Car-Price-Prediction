# Russian Car Plate Price Prediction with XGBoost

## Project Overview

This project tackles the challenge of predicting the market price of Russian vehicle registration plates. These plates, consisting of specific letter combinations, digits, and region codes, can have significant value based on their perceived prestige, government affiliation, or symbolic meaning.

The goal is to develop a machine learning model that accurately estimates these prices using historical sales data. The primary evaluation metric for this task is the **Symmetric Mean Absolute Percentage Error (SMAPE)**.

This repository contains the Python code and methodology used to build an XGBoost-based predictive model.

## Approach

Our approach involves several key stages:

1.  **Data Loading & Initial Setup**:
    * Loading training and testing datasets.
    * Combining datasets for consistent preprocessing.
    * Calculating global statistics (mean/median prices) for robust feature engineering.

2.  **Extensive Feature Engineering**:
    * **Plate Parsing**: Deconstructing the plate string into its core components (letters, numbers, region code, first letter, last two letters).
    * **Date Features**: Extracting year, month, day, day of the week, cyclical date features (sin/cos transformations), and flags for weekends/holidays.
    * **Governmental & Regional Information**: Leveraging supplemental data to identify government-affiliated plates, their significance, and mapping region codes to names.
    * **Prestige & Pattern Features**: Creating features based on desirable letter/number combinations (e.g., "AAA", "777"), repeated characters, sequential numbers, palindromic numbers, and a composite `prestige_score`.
    * **Encoding Techniques**:
        * Frequency encoding for the numerical part of plates.
        * Target encoding (mean log-price) for `letters`, `first_letter`, `last_letters`, and `region_name`.
    * **Interaction & Contextual Features**: Generating features like `letters_region_freq`, `is_gov_and_prestige`, lagged prices (`price_lag_1`), and `plate_listing_count`.
    * **Textual Features**: Using `CountVectorizer` on plate letters to capture character n-gram patterns.

3.  **Data Preprocessing**:
    * Identifying numerical and categorical features.
    * Applying `SimpleImputer` and `StandardScaler` to numerical features.
    * Applying `SimpleImputer` and `OrdinalEncoder` to categorical features.
    * Using a `ColumnTransformer` to manage these preprocessing steps.

4.  **XGBoost Model Training**:
    * Employing `StratifiedKFold` cross-validation (with target binning) for robust evaluation.
    * Training an XGBoost regressor (`XGBRegressor`) with parameters optimized for the Tweedie objective (`reg:tweedie`), suitable for price data.
    * Utilizing early stopping to prevent overfitting.

5.  **Prediction & Submission**:
    * Averaging predictions from all cross-validation folds for the test set.
    * Transforming predictions back from log scale to the original price scale.
    * Generating a `submission.csv` file in the format required by the competition.

## Key Technologies

* Python 3
* Pandas, NumPy
* Scikit-learn (for preprocessing, CV, metrics)
* XGBoost
* Holidays (for Russian holiday data)

## How to Run

1.  Ensure you have Python and the necessary libraries installed (see `requirements.txt` - *you'll need to create this file*).
2.  Place the `train.csv`, `test.csv`, and `supplemental_english.py` files in the appropriate directory (e.g., an `input` folder).
3.  Run the main Python script (e.g., `predict_plate_prices.py` - *you'll name your script*).
4.  The script will output a `submission_xgboost_enhanced_v5.csv` file.

## Results

The model achieves an average CV SMAPE of \[**Insert Your Best SMAPE Score Here, e.g., 35.XX%**\] on the validation sets, demonstrating a strong predictive capability for this complex regression task.

*(Optional: You can add a link to your Medium blog post here)*
