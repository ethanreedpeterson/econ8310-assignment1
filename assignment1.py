
# necessary imports
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES


# training and testing data
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_url  = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

train = pd.read_csv(train_url)
test  = pd.read_csv(test_url)


# Identify the target (dependent) variable for forecasting
def get_target(df: pd.DataFrame) -> str:
    
    # Prefer the 'trips' column if it exists
    if "trips" in df.columns:
        return "trips"
    
    # Otherwise, select the numeric column with highest variance
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not num_cols:
        raise ValueError("No numeric columns available for target.")
        
    return df[num_cols].var().sort_values(ascending=False).index[0]


# Extract the target time series from training data
y_col = get_target(train)
y = train[y_col].astype(float).to_numpy()


# defining ExponentialSmoothing model
model = ES(
    y,
    trend = "add",
    seasonal = "add",
    seasonal_periods = 168,
    initialization_method = "estimated"
)


# Fitting model to training data
modelFit = model.fit(optimized=True, use_brute=True)

# Generate forecasts for the next 744 hours (January test period)
pred = np.asarray(modelFit.forecast(744), dtype = float)

pred = np.clip(pred, 0, None)