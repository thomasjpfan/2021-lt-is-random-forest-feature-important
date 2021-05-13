+++
title = "Are Random Forest Feature Importances Useful?"
outputs = ["Reveal"]
+++

# Are Random Forest Feature Importances Useful?
Thomas J. Fan

{{< social >}}
{{< talk-link 2021-lt-is-random-forest-feature-important >}}

---

# Sometimes 🤔

---

## Load A Sample Dataset

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

diabetes = load_diabetes(as_frame=True)
X, y = diabetes['data'], diabetes['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)
```

---

## What does that dataset look like?

|    |         age |        sex |        bmi |          bp |
|---:|------------:|-----------:|-----------:|------------:|
|  0 |  0.0380759  |  0.0506801 |  0.0616962 |  0.0218724  |
|  1 | -0.00188202 | -0.0446416 | -0.0514741 | -0.0263278  |
|  2 |  0.0852989  |  0.0506801 |  0.0444512 | -0.00567061 |
|  3 | -0.0890629  | -0.0446416 | -0.011595  | -0.0366564  |
|  4 |  0.00538306 | -0.0446416 | -0.0363847 |  0.0218724  |

---

## Training Random Forest Regressors!

```python{1-4|6-7}
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

rf_base = RandomForestRegressor(random_state=0).fit(X_train, y_train)

mean_squared_error(y_test, rf_base.predict(X_test))
# 4000.6
```

---

### How to get feature importances

```python
rf_base.feature_importances_
# array([0.06140494, 0.01331342, 0.29823963, 0.09571546, 0.04687954,
#        0.05390033, 0.05259927, 0.02711794, 0.27920886, 0.0716206 ])
```

---

{{< figure src="images/impurity.png" height="640px" >}}

---

## Insert some random integer data

```python{1-2|4-5|7-8}
from numpy.random import default_rng
rng = default_rng(100)

X_random_int = X_train.assign(
    random_int=rng.integers(0, high=2, size=X_train.shape[0]))

X_random_int.loc[:4, "random_int"].to_numpy()
# array([1, 1, 0, 1, 0])
```

---

## Train a 🎄🎄🎄🎄

```python
from sklearn.ensemble import RandomForestRegressor
rf_random_int = (RandomForestRegressor(random_state=0)
                 .fit(X_random_int, y_train))
```


---

## Will the random integer be important for 🎄🎄🎄🎄?

{{% grid middle center %}}

{{< g 1 success>}}
Please Y for *Yes*
{{< /g >}}

{{< g 1 alert>}}
Please N for *No*
{{< /g >}}

{{% /grid %}}

---

{{< figure src="images/impurity_random_int.png" height="640px" >}}

---

## Insert some random uniform data

```python{1-2|4-5}
X_uniform_random = X_train.assign(
    random_uniform=rng.random(size=X_train.shape[0]))

X_uniform_random.loc[:4, "random_uniform"].to_numpy()
# array([0.34703067, 0.23571476, 0.20958332, 0.34343029, 0.40482601])
```

---

## Train a 🎄🎄🎄🎄

```python
rf_uniform_random = (RandomForestRegressor(random_state=0)
                    .fit(X_uniform_random, y))
```

---

## Will the random uniform feature be important for 🎄🎄🎄🎄?

{{% grid middle center %}}

{{< g 1 success>}}
Please 1 for *Yes*
{{< /g >}}

{{< g 1 alert>}}
Please 0 for *No*
{{< /g >}}

{{% /grid %}}

---

{{< figure src="images/impurity_very_random.png" height="640px" >}}

---

# Does SHAP values do better for random uniform features? 💡

---

## Nope

{{< figure src="images/shap_uniform_random.png" height="640px" >}}

---

# Are Random Forest Feature Importances Useful?

### **Sometimes** 🤔
