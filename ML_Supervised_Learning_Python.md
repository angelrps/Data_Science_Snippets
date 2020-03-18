# 0. Split Data
Create Training Set and Test Set with sklearn
```python
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
from sklearn.metrics import mean_absolute_error

# Use against predictions
mean_absolute_error(reg.predict(X_test), y_test)
```

**Calculated manually with numpy**
```python
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
import numpy as np

myMAPE = np.mean(np.abs(reg.predict(X_test) - y_test)/y_test)
```

### 2.1.3. RMSE (Root Mean Squared Error)
Measures average magnitud of errors.<br />
Units are the same as the target variable.<br />
Value range from 0 to infinite.<br />
Lower values are better.

```python
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

# 3. Cross Validation Score
Get more robust metrics using Cross Validation

```python
from sklearn.model_selection import cross_val_score

cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
# model = selected model
# cv= number of training/test sets
# X = inputs
# y = output
# scoring = scoring function
```

# 4. Testing Parameters
## 4.1. GridSearchCV
Search over a grid of parameters to find the best for your model. It returns an instance of the best model found. After this you need to train the model.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

reg_test = GridSearchCV(KNeighborsRegressor(),
                       param_grid={"n_neighbors":np.arange(3,50)},
                       cv = 5,
                       scoring = "neg_mean_absolute_error")
                       
# KNeighborsRegressor(): model I want to test
# param_grid = dictionary with parameters and values needed for the grid. These are model specific
# scoring = scoring function to evaluate
                       
# Train the model with the dataset
reg_test.fit(X,y)

# Get valuable info from the model
reg_test.best_score_
reg_test.best_estimator_
reg_test.best_params_
```
## 4.2. RandomizedSearchCV
If you want to find parameters itereating through a big grid your computer may crash. RandomizedSearchCV picks up a limited number of parameters randomly from your grid.

```python
from sklearn.model_selection import RandomizedSearchCV

reg_dt_test = RandomizedSearchCV(DecisionTreeRegressor(),
                                param_distributions={"max_depth":[2,3,5,8,10],
                                           "min_samples_leaf":[5,10,15,20,30,40]},
                                cv = 5,
                                scoring="neg_mean_absolute_error",
                                n_iter=5)

# n_iter = how many random values I want to try from param_distributions

# Train the model with the dataset
reg_dt_test.fit(X,y)
```
