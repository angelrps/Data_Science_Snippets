# 0. Split Data
Create Training Set and Test Set with sklearn
```python
# Load the library
from sklearn.model_selection import train_test_split

# Split data set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10)
```

# 1. Models
## 1.1. Model: Regression
Input variables (**X**) must be pandas **Data Frame** <br />
Output variable: (**y**) must be pandas **Series**
### 1.1.1. Linear Regression
```python
# Load the library
from sklearn.linear_model import LinearRegression

# Create an instance of the model
reg = LinearRegression()

# Train the regressor
reg.fit(X_train, y_train)

# Do predictions
reg.predict([[2540],[3500],[4000]])

```
### 1.1.2. (KNN) K Nearest Neighbor Regressor
**Parameters:**<br />
**k**: number of neighbors <br />
**weight**: way to give more importance to points which are nearby and less weight to the points which are farther away. <br />
- **Uniform**: all the same distance.<br />
- **Distance**: weighted average per distance <br />
- **Custom**: weighted average provided by user

```python
# Load the library 
from sklearn.neighbors import KNeighborsRegressor

# Create an instance.
# Define number of neighbors.
# weights possible values: 'uniform', 'distance', [callable] user defined function
regk = KNeighborsRegressor(n_neighbors=2, weights = 'uniform')

# Train the data
regk.fit(X_train, y_train)
```

### 1.1.3. Decision Tree Regressor

```python
from sklearn.tree import DecisionTreeRegressor
```
## 1.2. Model: Classification
Output is a Class

### 1.2.1. Logistic Regression
### 1.2.2. K Neighbor Classifier
### 1.2.3. Support Vector Machine
### 1.2.4. Decision Tree Classifier


# 2. Metrics
## 2.1. Metrics: Regression
### 2.1.1. MAE (Mean Absolute Error)
Measures average magnitud of the errors without considering their direction (all errors in absolute value).
It is intuitive to calculate, but you lose information related to the magnitud of th error.<br />
Units are the same as the target variable.<br />
Value range from 0 to infinite.<br />
Lower values are better.

**Calculated with sklearn**
```python
# Load the scorer
from sklearn.metrics import mean_absolute_error

# Use against predictions
mean_absolute_error(reg.predict(X_test), y_test)
```

**Calculated manually with numpy**
```python
# Load the library
import numpy as np

# reg.predict(X_test) is the prediction
myMAE = np.mean(np.abs(reg.predict(X_test) - y_test))
```

### 2.1.2. MAPE (Mean Absolute Percentage Error)
Similar to MAE but it measures the error in percentage.<br />
Value range from 0 to 100.<br />
Lower values are better.<br />
**MAPE is not in sklearn so we calculate it MANUALLY with pandas**
```python
# Load the library
import numpy as np

myMAPE = np.mean(np.abs(reg.predict(X_test) - y_test)/y_test)
```

### 2.1.3. RMSE (Root Mean Squared Error)
Measures average magnitud of errors.<br />
Units are the same as the target variable.<br />
Value range from 0 to infinite.<br />
Lower values are better.

```python
# Load the scorer
from sklearn.metrics import mean_squared_error

# Use against predictions (we have to calculate the squared root of MSE)
np.sqrt(mean_absolute_error(reg.predict(X_test), y_test))
```

| NOTES: MAE and RMSE |
| ------------------- |
| The square of RMSE minimizes errors < 1 and maximizes errors > 1. Meaning that if I have a moderate MAE but a big RMSE, there are a few points differing much from the prediction.
MAE and RMSE are different magnitudes and we MUST calculate both. |

### 2.1.3. Correlation and Bias

## 2.2. Metrics: Classification
### 2.2.1. Accuracy
### 2.2.2. Precision and Recall
### 2.2.3. AUC Curve
