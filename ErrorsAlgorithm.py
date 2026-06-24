import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import root_mean_squared_error

y_true=np.array([3.0,-0.5,2.0,7.0,5.0])
y_pred=np.array([2.5,0.0,2.0,8.0,4.2])
mae_sklearn=mean_absolute_error(y_true, y_pred)
mse_sklearn=mean_squared_error(y_true, y_pred)
rsme_sklearn=np.sqrt(mse_sklearn)
print("---Scikit-Learn Result---")
print(f"Mean Absolute Error(MAE): {mae_sklearn:.4f}")
print(f"Mean Squared Error(RMSE): {mse_sklearn:.4f}")
print(f"root mean squared Error : {mse_sklearn:.4f}\n")
