import os, glob
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
path = kagglehub.dataset_download("bumba5341/advertisingcsv")
print("Downloaded to:", path)
csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
if not csvs:
    raise FileNotFoundError("No CSV found")
csv_path = csvs[0]
print("Using CSV:", csv_path)
df = pd.read_csv(csv_path)
df.columns = [c.strip().lower() for c in df.columns]
def first_present(names):
    for n in names:
        if n in df.columns:
            return n
    return None
tv = first_present(["tv"])
radio = first_present(["radio"])
news = first_present(["newspaper","news","newspaper_spend"])
sales = first_present(["sales","sales_yr","sales_value","target"])
if any(x is None for x in [tv, radio, sales]):
    raise ValueError(f"Expected TV, Radio, Sales columns. Found: {list(df.columns)}")
features = [c for c in [tv, radio, news] if c is not None]
X = df[features].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[sales], errors="coerce")
data = pd.concat([X, y], axis=1).dropna()
X, y = data[features], data[sales]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"R²: {r2_score(y_test, preds):.3f}")
print(f"MAE: {mean_absolute_error(y_test, preds):.3f}")
lin = model.named_steps["linearregression"]
coef = pd.Series(lin.coef_, index=features).sort_values(ascending=False)
print("Coefficients:\n", coef)
plt.figure(figsize=(8,5))
coef.plot(kind="bar")
plt.title("products")
plt.ylabel("sales")
plt.tight_layout()
plt.show()
example_vals = {f: float(X[f].median()) for f in features}
print("Example features:", example_vals)
print("Predicted Sales:", float(model.predict(pd.DataFrame([example_vals]))))
