
import os, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

BASE = "/mnt/data/insure-eu"
DATA = os.path.join(BASE, "data", "eu_kpis_synthetic.csv")
PLOTS = os.path.join(BASE, "plots")
REPORT = os.path.join(BASE, "report")

os.makedirs(PLOTS, exist_ok=True)
os.makedirs(REPORT, exist_ok=True)

df = pd.read_csv(DATA)

# ----- EDA -----
eda_stats = df.describe(include='all')
eda_stats.to_csv(os.path.join(REPORT, "eda_summary.csv"))

# Pairwise scatter
fig_scatter = px.scatter_matrix(
    df,
    dimensions=["purchasing_power_index","health_coverage_pct","aged_65plus_pct","gdp_per_capita_usd","life_cover_likelihood"],
    color="label_purchase",
    title="Pairwise KPI Relationships (colored by label)"
)
fig_scatter.update_traces(diagonal_visible=False)
fig_scatter.write_html(os.path.join(PLOTS,"pairwise_scatter.html"))

# Correlations

# Correlations
corr = df[["purchasing_power_index","health_coverage_pct","aged_65plus_pct","gdp_per_capita_usd","life_cover_likelihood"]].corr()
corr.to_csv(os.path.join(REPORT, "kpi_correlations.csv"))
import plotly.graph_objects as go
fig_corr = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.index,
    colorscale='Viridis'
))
fig_corr.update_layout(title="KPI Correlation Heatmap")
fig_corr.write_html(os.path.join(PLOTS,"corr_heatmap.html"))


# ----- Clustering -----
X = df[["purchasing_power_index","health_coverage_pct","aged_65plus_pct","gdp_per_capita_usd"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=7, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters

# PCA
pca = PCA(n_components=2, random_state=7)
proj = pca.fit_transform(X_scaled)
df["pc1"] = proj[:,0]
df["pc2"] = proj[:,1]
fig_pca = px.scatter(df, x="pc1", y="pc2", color="cluster", hover_name="country",
                     title="PCA Projection of EU Regions by KPIs (KMeans clusters)")
fig_pca.write_html(os.path.join(PLOTS,"pca_clusters.html"))

# ----- Classification -----
X_cls = np.column_stack([X_scaled, clusters])
y = df["label_purchase"].values
X_train, X_test, y_train, y_test = train_test_split(X_cls, y, test_size=0.3, random_state=13, stratify=y)

logit = LogisticRegression(max_iter=500, random_state=13)
logit.fit(X_train, y_train)
y_pred_l = logit.predict(X_test)
y_proba_l = logit.predict_proba(X_test)[:,1]
report_logit = classification_report(y_test, y_pred_l, output_dict=True)
auc_logit = roc_auc_score(y_test, y_proba_l)

rf = RandomForestClassifier(n_estimators=300, random_state=13)
rf.fit(X_train, y_train)
y_pred_r = rf.predict(X_test)
y_proba_r = rf.predict_proba(X_test)[:,1]
report_rf = classification_report(y_test, y_pred_r, output_dict=True)
auc_rf = roc_auc_score(y_test, y_proba_r)

metrics = {
    "logistic_regression": {"classification_report": report_logit, "roc_auc": auc_logit},
    "random_forest": {"classification_report": report_rf, "roc_auc": auc_rf},
    "pca_explained_variance_ratio": list(pca.explained_variance_ratio_)
}
with open(os.path.join(REPORT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# RF Feature importance
rf_importances = rf.feature_importances_
fi_df = pd.DataFrame({
    "feature": ["ppi_z","health_cov_z","aged_65plus_z","gdp_pc_z","cluster_id"],
    "importance": rf_importances
}).sort_values("importance", ascending=False)
fi_df.to_csv(os.path.join(REPORT, "rf_feature_importance.csv"), index=False)

fig_fi = px.bar(fi_df, x="feature", y="importance", title="Random Forest Feature Importance")
fig_fi.write_html(os.path.join(PLOTS,"rf_feature_importance.html"))

# Country-level summary
summary = df[["country","purchasing_power_index","health_coverage_pct","aged_65plus_pct","gdp_per_capita_usd","life_cover_likelihood","cluster"]].sort_values("life_cover_likelihood", ascending=False)
summary.to_csv(os.path.join(REPORT, "country_summary.csv"), index=False)

# ROC curve (RF)
from sklearn.metrics import roc_curve
fpr_rf, tpr_rf, thr = roc_curve(y_test, y_proba_r)
roc_df = pd.DataFrame({"fpr": fpr_rf, "tpr": tpr_rf})
fig_roc = px.area(roc_df, x="fpr", y="tpr", title=f"ROC Curve (Random Forest) AUC={auc_rf:.3f}")
fig_roc.write_html(os.path.join(PLOTS, "roc_random_forest.html"))

print("Pipeline complete.")
