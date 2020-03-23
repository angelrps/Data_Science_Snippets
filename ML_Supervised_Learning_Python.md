# Table of Contents
* [0_Train-Test Data Split](#0_train-test-data-split)
* [1_Models](#1_models)
  * [1_1_Regression](#1_1_Regression)
    * [1_1_1_Linear Regression](#1_1_1_Linear-Regression)
    * [1_1_2_K-Nearest Neighbor Regressor_KNN)](#1_1_2_K-Nearest-Neighbor-Regressor_KNN)
    * [1_1_3_Decision Tree Regressor](#1_1_3_Decision-Tree-Regressor)
  * [1_2_Classification](#1_2_Classification)
    * [1_2_1_Logistic Regression](#1_2_1_Logistic-Regression)
    * [1_2_2_K Nearest Neighbor Classifier_KNN](#1_2_2_K-Nearest-Neighbor-Classifier_KNN)  
    * [1_2_3_Support Vector Machine_SVM](#1_2_3_Support-Vector-Machine_SVM)
    * [1_2_4_Decision_Tree_Classifier](#1_2_4_Decision-Tree-Classifier)
* [2_Metrics](#2_Metrics)
  * [2_1_Regression](#2_1_Regression)
    * [2_1_1_MAE_Mean Absolute Error](#2_1_1_MAE_Mean-Absolute-Error)
    * [2_1_2_MAPE_Mean Absolute Percentage Error](#2_1_2_MAPE_Mean-Absolute-Percentage-Error)
    * [2_1_3_RMSE_Root Mean Squared Error](#2_1_3_RMSE_Root-Mean-Squared-Error)
    * [2_1_4_Correlation](#2_1_4_Correlation)
    * [2_1_5_Bias](#2_1_5_Bias)
    * [2_1_6_Variance](#2_1_6_Variance)
  * [2_2_Classification](#2_2_Classification)
    * [2_2_1_Accuracy](#2_2_1_Accuracy)
    * [2_2_2_Precision](#2_2_2_Precision)
    * [2_2_3_Recall or Sensitivity](#2_2_3_Recall-or-Sensitivity)
    * [2_2_4_F1 score](#2_2_4_F1-score)
    * [2_2_5_Classification Report](#2_2_5_Classification-Report)
    * [2_2_6_ROC Curve_Receiver Operating Characteristic Curve)](#2_2_6_ROC-Curve_Receiver-Operating-Characteristic-Curve))
    * [2_2_7_AUC_Area Under the Curve](#2_2_7_AUC_Area-Under-the-Curve)
    * [2_2_8_Confusion Matrix](#2_2_8_Confusion-Matrix)
* [3_Cross Validation Score](#3_Cross-Validation-Score)
* [4_Testing Parameters](#4_Testing-Parameters)
  * [4_1_GridSearchCV](#4_1_GridSearchCV)
  * [4_2_RandomizedSearchCV](#4_2_RandomizedSearchCV)
* [5_Ensemble Learning](#5_Ensemble-Learning)
  * [5_1_VotingClassifier](#5_1_VotingClassifier)
  * [5_2_Bagging or Bootstrap Aggregation](#5_2_Bagging-or-Bootstrap-Aggregation)
  * [5_3_Random Forest](#5_3_Random-Forest)  
  * [5_4_Gradient Boosting Tree](#5_4_Gradient-Boosting-Tree)
* [6_References](#6_References)
  
# 0_Train Test Data Split
Create Training Set and Test Set with sklearn.
```python
from sklearn.model_selection import train_test_split

# Split data set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10)
```

# 1_Models
## 1_1_Regression
Input variables (**X**) must be pandas **Data Frame** <br />
Output variable: (**y**) must be pandas **Series**
### 1_1_1_Linear Regression
```python
from sklearn.linear_model import LinearRegression

# Create an instance of the model
reg = LinearRegression()

# Train the regressor
reg.fit(X_train, y_train)

# Do predictions
reg.predict([[2540],[3500],[4000]])

```
### 1_1_2_K Nearest Neighbor Regressor_KNN
**Parameters:**<br />
- **k**: number of neighbors <br />
- **weight**: way to give more weight to points which are nearby and less weight to the points which are farther away. <br />
  * `'uniform'`: all the same weight.<br />
  * `'distance'`: weighted average per distance. <br />
  * `'Custom'`: weighted average provided by user

```python
from sklearn.neighbors import KNeighborsRegressor

# Create an instance.
# Define number of neighbors.
# weights possible values: 'uniform', 'distance', [callable] user defined function
regk = KNeighborsRegressor(n_neighbors=2, weights = 'uniform')

# Train the data
regk.fit(X_train, y_train)
```

### 1_1_3_Decision Tree Regressor
Simple to understand, interpret and vizualise.

**Parameters:**<br />
- **max_depth**: number of splits <br />
- **min_samples_leaf**: minimum number of samples for each split group

```python
from sklearn.tree import DecisionTreeRegressor

# Create an instance.
regd = DecisionTreeRegressor (max_depth = 3,
                              min_samples_leaf=20)

# Train the data
regd.fit(X_train, y_train)
```

## 1_2_Classification
Output is a Class.

### 1_2_1_Logistic Regression

```python
# Load the library
from sklearn.linear_model import LogisticRegression

# Create an instance of the classifier
clr=LogisticRegression()

# Train the data
clr.fit(X_train,y_train)
```
### 1_2_2_K Nearest Neighbor Classifier_KNN

**Parameters:**<br />
Same as K-Nearest Neighbor Regressor.
```python
# Load the library
from sklearn.neighbors import KNeighborsClassifier

# Create an instance of the classifier
# parameter weights = 'uniform' as default
regk = KNeighborsClassifier(n_neighbors=5)

# Train the data 
regk.fit(X_train, y_train)
```
### 1_2_3_Support Vector Machine_SVM
Tries to separate the classes by a line. This line can be straight, circle, and more.<br />
It is computationally expensive so we will use it as a last option if the other models fail.

**Parameters:**<br />
- **C**: error margins. Large values = smaller margins. Small values = larger margins.
- **kernel**: function that transforms the dimensional space so we can separate de data when there is a non-linear separation problem.
  * `'linear'`: line of separation.
  * `'poly'`: curved line of separation.
  * `'rbf'`: circle of separation.
- **degree**: degree of the polynomal kernel function (`'poly'`).

```python
from sklearn.svm import SVC

# Create an instance of the classifier
clf = SVC(kernel="linear",C=10)

# Train the data
clf.fit(X_train,y_train)
```

### 1_2_4_Decision Tree Classifier
Same Parameters as Decision Tree Regressor.
```python
from sklearn.tree import DecisionTreeClassifier
clft = DecisionTreeClassifier(min_samples_leaf=20, max_depth=10)

# Train the data
clft.fit(X_train,y_train)
```
**Using GridSearchCV**
```python
from sklearn.tree import DecisionTreeClassifier

clft = GridSearchCV(DecisionTreeClassifier(),
                  param_grid={"max_depth":np.arange(2,20), 
                              "min_samples_leaf":np.arange(20,100,20)},
                  cv=3,
                  scoring="accuracy")
# Train the data                
clft.fit(X,y)

clft.best_score_
clft.best_params_
```

# 2_Metrics
## 2_1_Regression
### 2_1_1_MAE_Mean Absolute Error
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

### 2_1_2_MAPE_Mean Absolute Percentage Error
Similar to MAE but it measures the error in percentage.<br />
Value range from 0 to 100. Lower values are better.<br />
**MAPE is not in sklearn so we calculate it MANUALLY with pandas**
```python
import numpy as np

myMAPE = np.mean(np.abs(reg.predict(X_test) - y_test)/y_test)
```

### 2_1_3_RMSE_Root Mean Squared Error
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

### 2_1_4_Correlation
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
### 2_1_5_Bias
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

### 2_1_6_Variance
Is the average of errors in predictions between two different data sets.

```python
import numpy as np

np.mean(reg.predict(X_train) - reg.predict(X_test))
```

| Bias-Variance Tradeoff|
| ------------------- |
|- **High Bias** means that we are not capturing the complexity of the problem (**underfitting**).|
|- **High Variance** meansthat we may be modelling the noise in the training set (**overfitting**). |

## 2_2_Classification
### 2_2_1_Accuracy
It mesasures the overall predicted accuracy of the model in percentage.<br />
It is calculated as `(True Positives + True Negatives)/(True Positives + True Negatives + False Positives + False Negatives)`

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, clr.predict(X_test))

# Or get a better Accuracy with Cross Validation
from sklearn.model_selection import cross_val_score

cross_val_score(clr,X,y,scoring="accuracy", cv=5).mean()
```
### 2_2_2_Precision
It is like Accuracy but it only looks at data that you predicted positive.<br />
It is calculated as `(True Positives)/(True Positives + False Positives)`

```python
from sklearn.metrics import precision_score

precision_score(y_test,clf.predict(X_test))

# With Croos Validation
from sklearn.model_selection import cross_val_score

cross_val_score(clf,X,y,scoring="precision").mean()
```
### 2_2_3_Recall or Sensitivity
Ability of a model to find all the relevant cases within a dataset.<br />
It is calculated as `(True Positives)/(True Positives + False Negatives)`

```python
from sklearn.metrics import recall_score

recall_score(y_test, clf.predict(X_test))

# With Croos Validation
from sklearn.model_selection import cross_val_score

cross_val_score(clf,X,y,scoring="recall").mean()
```
### 2_2_4_F1 score
Harmonic mean of Precision and Recall.<br />
Value range from 0 (worst) to 1 (best).<br />
It is calculated as `2*(Recall * Precision) / (Recall + Precision)`

```python
from sklearn.metrics import f1_score

f1_score(y_test, clf.predict(X_test))
```

### 2_2_5_Classification Report
Builds a report showing Precision, Recall and F1-score of our model.

```python
from sklearn.metrics import classification_report

print(classification_report(y_test,clf.predict(X_test)))
```
### 2_2_6_ROC Curve_Receiver Operating Characteristic Curve)
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
### 2_2_7_AUC_Area Under the Curve
Value range from 0 to 1. Higher values are better. However if your AUC is below 0.5, you could invert all the outputs of your classifier and get a better score, so you did something wrong.<br />
Once you have calculated the `roc_curve` from the point above:
```python
from sklearn.metrics import auc

auc(fp,tp)

# Or using Cross Validation
cross_val_score(clr,X,y,scoring="roc_auc", cv=5).mean()
```

### 2_2_8_Confusion Matrix
It is not a metric but it helps to see how distributed your predictions are.
```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, clf.predict(X_test))
```

# 3_Cross Validation Score
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

# 4_Testing Parameters
## 4_1_GridSearchCV
Search over a grid of parameters to find the best for your model. It returns an instance of the best model found. After this you need to train the model.

**Example with KNeighborsRegressor()**
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
**Example with Support Vector Machines**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

clf = GridSearchCV(SVC(kernel="poly",),
                  param_grid={"C":[1,10,100,1000,10000], "degree":[2,3,4,5]},
                  cv=3,
                  scoring="accuracy")

# Train the model with the dataset
clf.fit(X,y)
```
## 4_2_RandomizedSearchCV
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
# 5_Ensemble Learning
Combination of multiple models in order to improve predictive accuracy.<br />
Prevents overfitting.<br />
It can be used in both Regression and Classification.<br />
Random Forest and Gradient Boosting Tree are ensemble methods.

## 5_1_VotingClassifier
Combines multiple different models into a single model, which is (ideally) stronger than any of the individual models alone.
Example below of VotingClassifier created out of a DecisionTreeClassifier and LogisticRegression
```python
from sklearn.ensemble import VotingClassifier

# Specify voting classifiers as a list of (name, sub-estimator) 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

classifiers = [('dectree', DecisionTreeClassifier(min_samples_leaf=16, max_depth=4)),
                ('log', LogisticRegression())]

# Create VotingClassifier
clf = VotingClassifier(estimators = classifiers)
```
## 5_2_Bagging or Bootstrap Aggregation
It divides the dataset into subsets, sampling with replacement, fits a base estimator on each subset, and then aggregate their individual predictions.<br />

```python
from sklearn.ensemble import BaggingClassifier

# base_estimator = KNeighborsClassifier in this example
# n_estimators: number of subsets. We can start with 100, and if it improves the single model, increase to 500, 1000, ...
# oob_score (Out Of Bag): set to True for small data sets. (default value = False)
clf=BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=4),n_estimators=100,oob_score=True)

# Train the model
clf.fit(X,y)
```

## 5_3_Random Forest
Enhanced version of Bagging, using Decision Trees.
Sklearn has its own algorithm for Random Forest.
**Parameters:**<br />
**N_estimators**: number of trees in the forest
**max_depth**: number of splits <br />
**min_samples_leaf**: minimum number of samples for each split group

```python
from sklearn.ensemble import RandomForestClassifier

# n_jobs: to specify how many concurrent processes/threads should be used. For -1, all CPUs are used.
clf = RandomForestClassifier(max_depth=3,
                             min_samples_leaf=20,
                             n_estimators=100,
                            n_jobs=-1)
                            
# Train the mdoel
clf.fit(X,y)
```

## 5_4_Gradient Boosting Tree
**Parameters:**<br />
**N_estimators**: number of trees in the forest <br />
**learning_rate**: how much correction do I keep from the precious model. Small values (<= 0.1) lead to much better generalization error. <br />
**max_depth**: number of splits <br />
**min_samples_leaf**: minimum number of samples for each split group
```python
from sklearn.ensemble import GradientBoostingRegressor

regGBT = GradientBoostingRegressor(max_depth=4,
                                min_samples_leaf=20,
                                n_estimators=100,
                                learning_rate=0.1)
```
Example using GridSearchCV
```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

reg = GridSearchCV(GradientBoostingRegressor(n_estimators=50),
                  param_grid={"max_depth":np.arange(2,10),
                             "learning_rate":np.arange(1,10)/10},
                  scoring="neg_mean_absolute_error",
                  cv=5)
```

# 6_References
https://github.com/Beovulfo/snippets3/blob/master/ML_python.md<br />
https://examples.dask.org/ <br />
https://scikit-learn.org <br />
https://machinelearningmastery.com/configure-gradient-boosting-algorithm/


