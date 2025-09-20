# Function to compute RMSD
rmsd <- function(pred, target) {
  sqrt(mean((pred - target)^2))
}

# Read target values
target <- scan("target.txt", what = numeric())

# Collect predictions from 10 files into a matrix
pred_list <- lapply(1:10, function(i) scan(paste0("predictions_", i, ".txt"), what = numeric()))
pred_matrix <- do.call(cbind, pred_list)

# Average predictions across files
avg_pred <- rowMeans(pred_matrix)

# Compute RMSD
rmsd_val <- rmsd(avg_pred, target)

# Compute Pearson correlation
cor_val <- cor(avg_pred, target, method = "pearson")

# Print results
cat("RMSD:", rmsd_val, "\n")
cat("Pearson correlation:", cor_val, "\n")