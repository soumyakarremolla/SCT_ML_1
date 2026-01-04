# House Price Prediction - Linear Regression

A machine learning project that predicts house prices using linear regression based on property features from the House Prices dataset.

## Overview

This project implements a linear regression model to predict house sale prices based on three key features:
- **GrLivArea**: Square footage of the above-ground living area
- **BedroomAbvGr**: Number of bedrooms above ground
- **FullBath**: Number of full bathrooms

The model is trained on historical housing data and provides both model evaluation metrics and dynamic prediction capabilities.

## Features

✓ Linear regression model implementation using scikit-learn
✓ Automatic handling of missing values (median imputation)
✓ Train-test split for model validation (80-20 split)
✓ Model evaluation with Mean Squared Error (MSE) and R² Score
✓ Interactive user input for house price predictions
✓ Clear console output for model metrics and predictions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SCT_ML_1
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

The following Python packages are required:
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning models and metrics

See `requirements.txt` for specific versions.

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Load the training data from `house_price_regression/data/train.csv`
2. Train a linear regression model on the selected features
3. Evaluate the model performance on the test set
4. Display evaluation metrics (MSE and R² Score)
5. Prompt for user input to make predictions on new house properties

### Example Interaction

```
Loading training data...
Using features: ['GrLivArea', 'BedroomAbvGr', 'FullBath']
Splitting data into train and test sets...
Training linear regression model...

Model Evaluation on Test Set:
Mean Squared Error: 1234567890.12
R^2 Score: 0.7234

--- Predict House Price ---
Enter square footage (GrLivArea): 2500
Enter number of bedrooms (BedroomAbvGr): 4
Enter number of full bathrooms (FullBath): 2

Predicted House Price: $450,000.00
```

## Data Structure

```
SCT_ML_1/
├── main.py                           # Main script
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── data/                             # Data directory
└── house_price_regression/data/      # Training data location
    └── train.csv                     # Training dataset with features and target
```

## Model Details

### Features Used
1. **GrLivArea**: Above-ground living area (in square feet)
2. **BedroomAbvGr**: Number of bedrooms above ground level
3. **FullBath**: Number of full bathrooms

### Data Preprocessing
- Missing values are handled using median imputation
- Data is split into 80% training and 20% testing sets
- Random state is set to 42 for reproducibility

### Model Evaluation
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values
- **R² Score**: Indicates the proportion of variance in the target variable explained by the model (ranges from 0 to 1, higher is better)

## Notes

- Ensure the training data file exists at the specified path: `house_price_regression/data/train.csv`
- The model supports multiple predictions without retraining
- All input values are validated before prediction
- Error handling is in place for invalid user inputs

## Future Improvements

- Add more features for better predictions
- Implement cross-validation for more robust model evaluation
- Add polynomial regression or other advanced models
- Create a web interface for easier predictions
- Add model persistence (save/load trained models)

## License

This project is open source and available under the MIT License.