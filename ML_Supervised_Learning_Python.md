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
### 1.1.2. K Neighbor Regressor
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
### 2.1.1. RMSE (Root Mean Squared Error)
### 2.1.2. MAE (Mean Absolute Error)
Measures average magnitud of the errors without considering their direction (all errors in absolute value).
It is intuitive to calculate, but you lose information related to the magnitud of th error.<br />
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

### 2.1.3. MAPE (Mean Absolute Percentage Error)
Similar to MAE but it measures the error in percentage.<br />
Value range from 0 to 100.<br />
Lower values are better.
**MAPE is not in sklearn so we calculate it MANUALLY with pandas**
```python
# Load the library
import numpy as np

myMAPE = np.mean(np.abs(reg.predict(X_test) - y_test)/y_test)
```

### 2.1.3. Correlation and Bias

## 2.2. Metrics: Classification
### 2.2.1. Accuracy
### 2.2.2. Precision and Recall
### 2.2.3. AUC Curve
