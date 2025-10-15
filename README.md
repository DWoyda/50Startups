# Regression Models Comparison on 50_Startups Dataset

## Project Overview
This project analyzes the "50_Startups" dataset using multiple regression techniques to predict company profit based on expenditures in R&D, Administration, Marketing, and geographic location (State). The objective was to compare different regression approaches, automate preprocessing, and evaluate model performance.

**Main goals:**
- Build accurate regression models
- Compare multiple model types
- Automate preprocessing with pipelines
- Understand variable significance with statistical tools (OLS)

---

## Exploratory Data Analysis (EDA)

Before modeling, we thoroughly analyzed the data:

- No missing values or duplicates
- Normal distribution of target variable (`Profit`)
- Strong correlations found:
  - `R&D Spend` - most correlated with `Profit` (~0.97)
  - `Marketing Spend` - moderate correlation (~0.75)
  - `Administration` - weak correlation (~0.20)

**Visuals used:**
- Pairplot 
- Boxplot
- Correlation heatmap
- Distribution plots

**Interpretation:**
- R&D is clearly the most impactful feature.
- Marketing spend might help, but less so.
- Administration appears less relevant.

---

## Regression Approaches
To better understand the modeling process, we implemented **three approaches**:

### 1. Manual Regression with Backward Elimination (OLS)

- Step-by-step regression using `statsmodels.OLS`.
- Applied **backward elimination**:
  - Repeatedly removed the feature with the highest p-value.
- End result: **Only `R&D Spend` remained statistically significant**.

**Key Concepts Learned:**
- p-values and confidence intervals
- OLS summary interpretation
- Statistical feature selection, not just algorithmic

---

### 2. Pipeline with Single Regressor (`LinearRegression`)

Built using `Pipeline` and `ColumnTransformer`:

- Applied `StandardScaler` to numeric features
- Used `OneHotEncoder` (drop='first') for `State`
- Trained `LinearRegression` model

Tools used:
- `get_feature_names_out()` to inspect transformed features
- `sm.add_constant()` to include intercepts if needed

Benefit: Fully **modular and reusable**, great for production or deployment.

---

### 3. Pipeline for Multiple Models – Automated Comparison

Created a loop to compare models, scalers, and transformations:

Models compared:
- `Linear Regression`
- `Polynomial Regression`
- `Support Vector Regression (SVR)`
- `Decision Tree`
- `Random Forest`

Tested combinations of:
- `StandardScaler` / `MinMaxScaler`
- `OneHotEncoder`

Metrics captured:
- `MSE` (Mean Squared Error)
- `Train time (s)`

**Results Summary**:

| Model               | Scaler         | MSE           |
|--------------------|----------------|---------------|
| Decision Tree       | StandardScaler | 35,225,563.35 |
| Random Forest       | MinMaxScaler   | 38,318,888.21 |
| Linear Regression   | MinMaxScaler   | 83,502,864.03 |
| SVR                 | StandardScaler | 1,483,093,113 |

**Best performance**: Tree-based models  
**Worst**: SVR – unsuitable without heavy tuning or outlier handling

---

## 🧠 Why 3 Approaches?

Although one final automated pipeline might seem sufficient, we purposefully implemented all three for **learning and clarity**:

- 🔍 **Manual OLS** taught statistical interpretation and backward elimination.
- ⚙️ **Single-model pipeline** clarified how encoding/scaling affects modeling.
- 🤖 **Multiple-model pipeline** allowed for comprehensive model comparison.

🎯 **Value:** This structure helped **understand modeling from the ground up**, not just get a final prediction. Ideal for portfolio, interviews, and real-world prep.

---

## 📌 Key Conclusions

- `R&D Spend` is by far the **most important predictor** (confirmed both statistically and algorithmically).
- **Random Forest** and **Decision Tree** achieved the lowest MSE – robust for small tabular datasets.
- **SVR** performs poorly without tuning – it's not plug-and-play.
- **OLS** adds deep interpretability missing from typical ML pipelines.
- Building **modular pipelines** makes the code reusable and scalable.

---











