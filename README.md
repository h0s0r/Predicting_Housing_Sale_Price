___
___
# California Housing Price Prediction

A simple Linear Regression project to predict house prices using Python and scikit-learn.
___
## What This Does

This project:
- Loads California housing data
- Trains a model to predict house prices
- Shows how well the predictions match real prices
- Creates charts to visualize the data
___
## Requirements

Install these packages:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```
___
## How to Run

```bash
python main.py
```
___
## Files

- `main.py` - The main code that trains the model
- `california_housing.csv` - Dataset file (created when you run the code)
___
## Dataset Features

The model uses these inputs to predict house prices:
- Median income of the area
- House age
- Average rooms and bedrooms
- Population and occupancy
- Location (latitude/longitude)
___
## Results

The model achieves:
- **R² Score**: 0.575 (means the model is 57.5% accurate)
- **MSE**: 0.555 (lower is better)
___
## Visualizations Created

1. **Feature Distributions** - Shows spread of each feature
2. **Correlation Heatmap** - Shows which features affect price most
3. **Actual vs Predicted** - Compares real prices with predictions
4. **Residual Plot** - Shows prediction errors
___
## What I Learned

- How to load and explore datasets
- Training a machine learning model
- Evaluating model performance
- Creating visualizations to understand data
___
# The End - By h0s0r
___
___