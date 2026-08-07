# Ex.No: 6               HOLT WINTERS METHOD

### AIM: 

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv(
    "/content/index_1.csv",
    parse_dates=["date"],
    index_col="date"
)

data_monthly = data["money"].resample("MS").sum()

print(data_monthly.head())

data_monthly.plot(figsize=(10, 5))
plt.show()

scaler = MinMaxScaler()

scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)

scaled_data.plot(figsize=(10, 5))
plt.show()

if len(data_monthly) >= 12:
    decomposition = seasonal_decompose(
        data_monthly,
        model="additive",
        period=6
    )
    decomposition.plot()
    plt.show()

scaled_data = scaled_data + 1

train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

model_add = ExponentialSmoothing(
    train_data,
    trend="add",
    seasonal="mul",
    seasonal_periods=4,
    initialization_method="estimated"
).fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)

ax.legend(["train_data", "test_predictions_add", "test_data"])
ax.set_title("Visual evaluation")

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))

print(f"RMSE for test predictions: {rmse}")
print(f"Standard Deviation of scaled data: {scaled_data.std()}")
print(f"Mean of scaled data: {scaled_data.mean()}")

data_monthly_positive = data_monthly.replace(0, np.finfo(float).eps)

final_model = ExponentialSmoothing(
    data_monthly_positive,
    trend="add",
    seasonal="mul",
    seasonal_periods=4,
    initialization_method="estimated"
).fit()

final_predictions = final_model.forecast(steps=max(1, int(len(data_monthly) / 4)))

ax = data_monthly.plot()
final_predictions.plot(ax=ax)

ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel("Date")
ax.set_ylabel("Money")
ax.set_title("Prediction")

plt.show()
```
### OUTPUT:


<img width="877" height="461" alt="image" src="https://github.com/user-attachments/assets/5d2a2efb-6a05-4b73-95ec-13839218c5cc" />


<img width="685" height="457" alt="image" src="https://github.com/user-attachments/assets/53e316b2-432a-4d51-9e63-abb2d1d09375" />

<img width="667" height="477" alt="image" src="https://github.com/user-attachments/assets/b3ecdf68-18bd-42e2-8006-be223e727b86" />

<img width="657" height="456" alt="image" src="https://github.com/user-attachments/assets/6830d7c6-2d53-4ed6-90c2-1cc4cb2909ce" />

<img width="1227" height="117" alt="image" src="https://github.com/user-attachments/assets/b47de1e3-2f20-4835-ae03-01bf02a1672c" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
