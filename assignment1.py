import numpy as np
import pandas as pd
from pygam import LinearGAM, s, f

train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
train = pd.read_csv(train_url)

y = train["trips"].astype(float).to_numpy()
n = len(y)
H = 744

t = np.arange(n)
hour = (t % 24)
dow  = ((t // 24) % 7).astype(int)

hour_rad = 2 * np.pi * hour / 24.0
hour_sin = np.sin(hour_rad)
hour_cos = np.cos(hour_rad)

X = np.column_stack([hour_sin, hour_cos, dow, t])

model = LinearGAM(
    s(0, n_splines=20) +
    s(1, n_splines=20) +
    f(2) +
    s(3, n_splines=80)
)

lam_grid = np.logspace(-3, 3, 9)
modelFit = model.gridsearch(X, y, lam=lam_grid, progress=False)

t_future = np.arange(n, n + H)
hour_f = (t_future % 24)
dow_f  = ((t_future // 24) % 7).astype(int)

hour_rad_f = 2 * np.pi * hour_f / 24.0
X_future = np.column_stack([
    np.sin(hour_rad_f),
    np.cos(hour_rad_f),
    dow_f,
    t_future
])

pred = modelFit.predict(X_future).astype(float)
pred = np.clip(pred, 0, None)
