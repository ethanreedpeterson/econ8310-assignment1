# Generalized Additive Model route!
import numpy as np
import pandas as pd
from pygam import LinearGAM, s, f

# data loading
train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_url  = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

train = pd.read_csv(train_url)
test  = pd.read_csv(test_url)


X_train = train[["hour", "day", "month"]].values
y_train = train["trips"].values

X_test = test[["hour", "day", "month"]].values

# actual model
model = LinearGAM(
    s(0, n_splines=24) +
    s(1, n_splines=31) +
    f(2)
)

modelFit = model.fit(X_train, y_train)
pred = modelFit.predict(X_test)

# output of prediction
forecast_df = test.copy()
forecast_df["predicted_trips"] = pred
forecast_df.head()