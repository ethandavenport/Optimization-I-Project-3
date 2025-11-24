# **OPT PROJECT 3 REPORT**

## **Price-Sensitive Newsvendor Optimization with Rush & Disposal Costs**

### **By: `Abhay Puri`, `Abhiroop Kumar`, `Ethan Davenport`, `Liam Thompson`**

---

# **1. Overview**

This project extends the traditional Newsvendor (NV) framework to incorporate **price-dependent demand**, **rush printing**, and **disposal costs**.
Using the dataset provided, we:

1. Estimated the **price–demand relationship** using linear regression.
2. Constructed an **LP for fixed-price printing** (p = 1).
3. Developed a **Quadratic Program (QP)** to simultaneously optimize **price** and **quantity**.
4. Used **4,000 bootstrap iterations** to evaluate the robustness of the optimal decisions.

---

# **2. Problem Description**

A publishing company must decide how many units of a book to print before uncertain demand is realized. Unlike the classic NV model, this environment includes:

### **Operational Realities**

* **Rush printing** at cost (g = 0.75) if demand exceeds initial print quantity.
* **Disposal cost** (t = 0.15) for unsold units.

### **Price Sensitivity**

Demand depends on price:

$$[
D(p) = a + b p + \varepsilon
]$$

Thus, price is no longer a fixed parameter—it becomes a **decision variable**.

Our objective is to:

* Fit a statistical model for demand
* Optimize print quantity at fixed price (NV baseline)
* Optimize price and quantity jointly (QP)
* Conduct bootstrap analysis for robustness
* Compare approaches for a managerial decision recommendation

---

# **3. Data Description**

The dataset contains historical observations of:

* Daily **price**
* Corresponding **demand**

We use:

```python
X = df[["price"]].to_numpy()
y = df["demand"].to_numpy()
```

We fit:

```python
ols = LinearRegression()
ols.fit(X, y)

a = float(ols.intercept_)
b = float(ols.coef_[0])
r2 = float(ols.score(X, y))
residuals = y - (a + b*X[:,0])
```

### Summary:

* **Intercept** (a) captures baseline demand
* **Slope** (b < 0) indicates strong negative price elasticity
* **Residuals** represent unexplained shocks and are used to build empirical demand scenarios

---

# **4. Model Formulation**

## **4.1 Fixed Price Model (p = 1)**

Demand at p=1:

```python
d1 = a + b*p_fixed + residuals
```

Profit per scenario:

$$[
\pi_i(q)= pD_i - qc - g(D_i-q)^+ - t(q-D_i)^+
]$$

Decision variable:

* (q) = quantity to print

Auxiliary variables:

* `r[i]` = rush
* `d[i]` = disposal
* `s[i]` used in constraints

LP structure (from notebook):

```python
m = gp.Model()
q = m.addMVar(1, lb=0.0, name="q")
s = m.addMVar(n, lb=0.0, name="s")
r = m.addMVar(n, lb=0.0, name="r")
d = m.addMVar(n, lb=0.0, name="d")

m.addConstr(s <= q[0])
m.addConstr(s <= d1)
m.addConstr(r >= d1 - q[0])
m.addConstr(d >= q[0] - d1)
```

Objective:

```python
total_revenue = p_fixed * np.sum(d1)
cost_expr = c * n * q[0] + g * quicksum(r) + t * quicksum(d)
obj = (total_revenue - cost_expr) / n
```

---

## **4.2 Joint Price + Quantity Model (QP)**

Demand now depends on decision (p):

$$[
D_i(p) = a + bp + \varepsilon_i
]$$

Revenue is:

$$[
p D_i(p) = p(a + bp + \varepsilon_i)
]$$

Introducing quadratic terms → QP.

Model in notebook:

```python
mq = gp.Model()

p = mq.addMVar(1, lb=0.0, name="p")
q = mq.addMVar(1, lb=0.0, name="q")

s = mq.addMVar(n, lb=0.0, name="s")
r = mq.addMVar(n, lb=0.0, name="r")
d = mq.addMVar(n, lb=0.0, name="d")

d2 = a + b * p[0] + residuals

mq.addConstr(s <= q[0])
mq.addConstr(s <= d2)
mq.addConstr(r >= d2 - q[0])
mq.addConstr(d >= q[0] - d2)

revenue_qp = p[0] * gp.quicksum(d2)
cost_qp = c * n * q[0] + g * quicksum(r) + t * quicksum(d)
obj_qp = (revenue_qp - cost_qp) / n
```

---

# **5. Optimization Results**

## **5.1 Fixed Price LP**

| Metric           | Value        |
| ---------------- | ------------ |
| Optimal quantity | **≈ 471.87** |
| Expected profit  | **≈ 231.48** |

Interpretation:

* With price fixed, the NV-style model chooses a moderate quantity to balance rush vs. disposal costs.
* Profit is limited because price is not optimized.

---

## **5.2 Joint Price + Quantity QP**

| Metric           | Value        |
| ---------------- | ------------ |
| Optimal price    | **≈ 0.954**  |
| Optimal quantity | **≈ 535.29** |
| Expected profit  | **≈ 234.42** |

Interpretation:

* Slightly lowering the price increases demand substantially (due to the steep slope (b)).
* Higher demand justifies a larger print run.
* Profit increases by ~1.2–1.5%.

This aligns with economic intuition:
**Lower price → higher demand → better matching of fixed operational costs.**

---

# **6. Bootstrap Sensitivity Analysis**

Purpose:

* Evaluate how sensitive optimal decisions are to noise in the underlying dataset
* Test model robustness under demand shocks

Bootstrap loop (from notebook):

```python
results = boot(4000, df)
```

Where `boot()` internally:

* Resamples the data
* Refits regression
* Rebuilds QP
* Solves for p*, q*, profit
* Stores results

Mean results:

| Metric   | Mean Bootstrap Value |
| -------- | -------------------- |
| Price    | **0.9547**           |
| Quantity | **534.49**           |
| Profit   | **234.66**           |

### Interpretation:

* Very small variation in optimal decisions
* Profit distribution is tight (low risk)
* The model is **highly stable** under resampling uncertainty

Your notebook already includes visualizations for:

* Histogram of bootstrap prices
* Histogram of bootstrap quantities
* Histogram of profits
* Joint distribution of price & quantity (scatter + histograms)

These figures should be included in the report PDF to satisfy the professor’s requirement for “informative visualizations.”

---

# **7. Managerial Insights & Recommendations (Q8)**

### **Is your boss’s NV model as good as this one?**

**Not exactly.**
The NV model is simple and intuitive but assumes:

* Price does not affect demand
* Only quantity matters
* No rush/disposal structure unless manually added

This causes the NV model to underperform on datasets where price elasticity is significant—which is true here.

### **Does switching models improve revenue?**

**Yes.**

* Fixed-price model: 231.48
* Price-sensitive model: 234.42
* Difference: **+2.94 profit units**

Even small improvements compound across many titles.

### **Advantages of Standard NV**

* Very simple
* Fast to compute
* Requires minimal data
* Works when price is fixed

### **Disadvantages**

* Ignores price elasticity
* Ignores rush/disposal unless extended
* Lower expected profit

### **Advantages of Extended QP Model**

* Incorporates real price–demand behavior
* Models operational realities (rush, disposal)
* Provides higher and more robust expected profit
* Bootstrap confirms stability

### **Disadvantages**

* More complex
* Requires regression + optimization tools
* Needs more data

---

# **8. Final Recommendation**

To the publishing operations team:

* For **high-volume or high-margin titles**, use the **extended QP model**.

  * It provides a **more accurate operational picture**,
  * Produces **higher expected profit**,
  * And does so **consistently** across bootstrap resamples.

* The NV model may still be used as a **quick heuristic** or when:

  * Price is externally fixed
  * Data availability is limited
  * Fast, approximate decisions are acceptable

Given the results, for this title the extended model is the **preferred decision-making tool**.

---

# END OF REPORT