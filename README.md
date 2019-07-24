[![codecov](https://codecov.io/gh/tgsmith61591/skoot/branch/master/graph/badge.svg)](https://codecov.io/gh/tgsmith61591/skoot)
![Supported versions](https://img.shields.io/badge/python-3.5-blue.svg)
![Supported versions](https://img.shields.io/badge/python-3.6-blue.svg)
![Supported versions](https://img.shields.io/badge/python-3.7-blue.svg)
[![CircleCI](https://circleci.com/gh/tgsmith61591/skoot.svg?style=svg)](https://circleci.com/gh/tgsmith61591/skoot)

# skoot

Skoot is a lightweight python library of machine learning transformer classes 
that interact with [scikit-learn](https://github.com/scikit-learn/scikit-learn)
and [pandas](https://github.com/pandas-dev/pandas). 
Its objective is to expedite data munging and pre-processing tasks that can
tend to take up so much of data science practitioners' time. See 
[the documentation](https://tgsmith61591.github.io/skoot) for more info.

__Note that skoot is the preferred 
alternative to the now deprecated [skutil](https://github.com/tgsmith61591/skutil) 
library__

## Two minutes to model-readiness

Real world data is nasty. Most data scientists spend the majority of their time
tackling data cleansing tasks. With skoot, we can automate away so much of the
bespoke hacking solutions that consume data scientists' time. 

In this example, we'll examine a common dataset (the 
[adult dataset](https://archive.ics.uci.edu/ml/datasets/Adult) from the UCI 
machine learning repo) that requires significant pre-processing.

```python
from skoot.datasets import load_adult_df
from skoot.feature_selection import FeatureFilter
from skoot.decomposition import SelectivePCA
from skoot.preprocessing import DummyEncoder
from skoot.utils.dataframe import get_numeric_columns
from skoot.utils.dataframe import get_categorical_columns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load the dataset with the skoot-native loader & split it
adult = load_adult_df(tgt_name="target")
y = adult.pop("target")
X_train, X_test, y_train, y_test = train_test_split(
    adult, y, random_state=42, test_size=0.2)
    
# get numeric and categorical feature names
num_cols = get_numeric_columns(X_train).columns
obj_cols = get_categorical_columns(X_train).columns

# remove the education-num from the num_cols since we're going to remove it
num_cols = num_cols[~(num_cols == "education-num")]
    
# build a pipeline
pipe = Pipeline([
    # drop out the ordinal level that's otherwise equal to "education"
    ("dropper", FeatureFilter(cols=["education-num"])),
    
    # decompose the numeric features with PCA
    ("pca", SelectivePCA(cols=num_cols)),
    
    # dummy encode the categorical features
    ("dummy", DummyEncoder(cols=obj_cols, handle_unknown="ignore")),
    
    # and a simple classifier class
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

pipe.fit(X_train, y_train)

# produce predictions
preds = pipe.predict(X_test)
print("Test accuracy: %.3f" % accuracy_score(y_test, preds))
```

For more tutorials, check out [the documentation](https://tgsmith61591.github.io/skoot).