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
### 1.1.2. (KNN) K-Nearest Neighbor Regressor
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
Simple to understand, interpret and vizualise.
**Parameters:**<br />
**max_depth**: number of splits <br />
**min_samples_leaf**: minimum number of samples for each split group

```python
from sklearn.tree import DecisionTreeRegressor

# Create an instance.
regd = DecisionTreeRegressor (max_depth = 3,
                              min_samples_leaf=20)

# Train the data
regd.fit(X_train, y_train)
```

## 1.2. Model: Classification
Output is a Class

### 1.2.1. Logistic Regression

```python
# Load the library
from sklearn.linear_model import LogisticRegression

# Create an instance of the classifier
clr=LogisticRegression()

# Train the data
clr.fit(X_train,y_train)
```
### 1.2.2. K-Nearest Neighbor Classifier
**Parameters:**<br />
Same as K-Nearest Neighbor Classifier
```python
# Load the library
from sklearn.neighbors import KNeighborsClassifier

# Create an instance of the classifier
# parameter weights = 'uniform' as default
regk = KNeighborsClassifier(n_neighbors=5)

# Train the data 
regk.fit(X_train, y_train)
```
### 1.2.3. Support Vector Machine
### 1.2.4. Decision Tree Classifier


# 2. Metrics
## 2.1. Metrics: Regression
### 2.1.1. MAE (Mean Absolute Error)
Measures average magnitud of the errors without considering their direction (all errors in absolute value).
It is intuitive to calculate, but you lose information related to the magnitud of th error.<br />
Units are the same as the target variable.<br />
Value range from 0 to infinite. Lower values are better.

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
Value range from 0 to 100. Lower values are better.<br />
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

### 2.1.4. Correlation
It measures whether or not there is a relationship between two variables. There should be a strong correlation between predictions and real values.

**With numpy**
```python
import numpy as np

# corrcoef() returns the correlation matrix.
# when applied to two values, the element [0][1] will give us the correlation coefficient we are looking for
# reg.predict(X_test) = predicted values

np.corrcoef(reg.predict(X_test), y_test)[0][1]
```

**With sklearn**

This function is not implemented in sklearn but we could create our own scorer and use it together with 'cross_val_score()'.
```python
import numpy as np
from sklearn.metrics import make_scorer

def mycorr(pred,y_test):
  return np.corrcoef(pred,y_test)[0][1]

cross_val_score(model,X,y,cv=5,scoring=make_scorer(mycorr)).mean()
```
### 2.1.5. Bias
It is the average of errors (prediction values minus real values).<br />
Negative errors will compensate positive ones.<br />

**With numpy**
```python
import numpy as np

# reg.predict(X_test) = predicted values

np.mean(reg.predict(X_test) - y_test)
```

**With sklearn**

This function is not implemented in sklearn but we could create our own scorer and use it together with 'cross_val_score()'.
```python
import numpy as np
from sklearn.metrics import make_scorer

def mybias(pred,y_test):
  return np.mean(pred - y_test)

cross_val_score(model,X,y,cv=5,scoring=make_scorer(mybias)).mean()
```

### 2.1.6. Variance
Is the average of errors in predictions between two different data sets.

```python
import numpy as np

np.mean(reg.predict(X_train) - reg.predict(X_test))
```

| Bias-Variance Tradeoff|
| ------------------- |
|- **High Bias** means that we are not capturing the complexity of the problem (**underfitting**).|
|- **High Variance** meansthat we may be modelling the noise in the training set (**overfitting**). |

## 2.2. Metrics: Classification
### 2.2.1. Accuracy
It mesasures the overall predicted accuracy of the model in percentage.<br />
It is calculated as `(True Positives + True Negatives)/(True Positives + True Negatives + False Positives + False Negatives)`

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, clr.predict(X_test))

# Or get a better Accuracy with Cross Validation
from sklearn.model_selection import cross_val_score

cross_val_score(clr,X,y,scoring="accuracy", cv=5).mean()
```
### 2.2.2. Precision
It is like Accuracy but it only looks at data that you predicted positive.<br />
It is calculated as `(True Positives)/(True Positives + False Positives)`

```python
from sklearn.metrics import precision_score

precision_score(y_test,clf.predict(X_test))

# With Croos Validation
from sklearn.model_selection import cross_val_score

cross_val_score(clf,X,y,scoring="precision").mean()
```
### 2.2.3. Recall (Sensitivity)
Ability of a model to find all the relevant cases within a dataset.<br />
It is calculated as `(True Positives)/(True Positives + False Negatives)`

```python
from sklearn.metrics import recall_score

recall_score(y_test, clf.predict(X_test))

# With Croos Validation
from sklearn.model_selection import cross_val_score

cross_val_score(clf,X,y,scoring="recall").mean()
```
### 2.2.4. F1 score
Harmonic mean of Precision and Recall.<br />
Value range from 0 (worst) to 1 (best).<br />
It is calculated as `2*(Recall * Precision) / (Recall + Precision)`

```python
from sklearn.metrics import f1_score

f1_score(y_test, clf.predict(X_test))
```

### 2.2.5. Classification Report
Builds a report showing Precision, Recall and F1-score of our model.

```python
from sklearn.metrics import classification_report

print(classification_report(y_test,clf.predict(X_test)))
```
### 2.2.6. ROC Curve (Receiver Operating Characteristic Curve)
It show how confident is your classifier with the area under the curve.

```python
from sklearn.metrics import roc_curve

# Get predictions in form of probabilities
pred = clf.best_estimator_.predict_proba(X_test)

# We chose the target
target_pos = 1 # Or 0 for the other class
fp,tp,_ = roc_curve(y_test,pred[:,target_pos])
plt.plot(fp,tp)
```
### 2.2.7. AUC (Area Under the Curve)
Value range from 0 to 1. Higher values are better. However if your AUC is below 0.5, you could invert all the outputs of your classifier and get a better score, so you did something wrong.<br />
Once you have calculated the `roc_curve` from the point above:
```python
from sklearn.metrics import auc

auc(fp,tp)

# Or using Cross Validation
cross_val_score(clr,X,y,scoring="roc_auc", cv=5).mean()
```

### * Confusion Matrix
It is not a metric but it helps to see how distributed your predictions are.
```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, clf.predict(X_test))
```

# 3. Cross Validation Score
Get more robust metrics using Cross Validation.<br />
It returns an array with all values. We can then calculate the mean of them.

```python
from sklearn.model_selection import cross_val_score

cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error").mean()
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
