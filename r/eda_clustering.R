
# EDA and Clustering in R (uses Plotly)
# Run: Rscript eda_clustering.R
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(plotly)
  library(ggplot2)
  library(stats)
  library(factoextra)
})

base_dir <- normalizePath(file.path(dirname(getwd()), ""))
data_path <- file.path(base_dir, "data", "eu_kpis_synthetic.csv")
plots_dir <- file.path(base_dir, "plots")
dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)

df <- read_csv(data_path, show_col_types = FALSE)

# Basic summary
summary_stats <- summary(df)
write.csv(as.data.frame(summary_stats), file.path(base_dir, "report", "r_eda_summary.csv"), row.names = TRUE)

# Numeric subset
num_cols <- c("purchasing_power_index","health_coverage_pct","aged_65plus_pct","gdp_per_capita_usd","life_cover_likelihood")
df_num <- df[, num_cols]

# KMeans (k=3)
set.seed(7)
km <- kmeans(scale(df[, c("purchasing_power_index","health_coverage_pct","aged_65plus_pct","gdp_per_capita_usd")]), centers=3, nstart=10)
df$cluster_r <- km$cluster

# 2D PCA for Plot
pca <- prcomp(scale(df[, c("purchasing_power_index","health_coverage_pct","aged_65plus_pct","gdp_per_capita_usd")]), center = TRUE, scale. = TRUE)
pca_df <- as.data.frame(pca$x[,1:2])
colnames(pca_df) <- c("PC1","PC2")
pca_df$country <- df$country
pca_df$cluster_r <- factor(df$cluster_r)

p <- plot_ly(pca_df, x = ~PC1, y = ~PC2, color = ~cluster_r, type = "scatter", mode = "markers",
             text = ~country, hoverinfo = "text") %>%
  layout(title = "R: PCA of EU KPIs with KMeans clusters")

htmlwidgets::saveWidget(as_widget(p), file.path(plots_dir, "r_pca_clusters.html"))
