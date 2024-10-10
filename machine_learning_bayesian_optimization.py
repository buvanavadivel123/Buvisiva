import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args

iris = load_iris()
x, y = iris.data, iris.target

space = [
    Integer(10,200,name = "n_estimators"),
    Integer(1,10,name = "max_depth"),
    Real(0.01, 1.0, name = "max_features"),
    Categorical(["gini","entropy"], name = "criterion")
]
@use_named_args(space)
def objective(**params):
    clf = RandomForestClassifier(**params, random_state=42)
    return -np.mean(cross_val_score(clf,x,y,cv=5,n_jobs=-1,scoring="accuracy"))

res = gp_minimize(objective, space, n_calls = 50, random_state = 42)
print(res)

print("n_estimators")
print(res.x[0])

print("max_depth")
print(res.x[1])

print("max_feature")
print(res.x[2])

print("criterion")
print(res.x[3])

param_grid = {
    "n_estimators" : res.x[0],
    "max_depth" : res.x[1],
    "max_features":res.x[2],
    "criterion" : res.x[3]
}

model = RandomForestClassifier(**param_grid, random_state = 42)
model.fit(x, y)

new_data = [[5.1,3.5,1.3,0.3],[6.2,3.4,5.3,2.2]]
y_prediciton = model.predict(new_data)
print(y_prediciton)