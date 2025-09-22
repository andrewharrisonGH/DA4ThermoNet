# Function to compute RMSD
rmsd <- function(pred, target) {
  sqrt(mean((pred - target)^2))
}

# Read target values
target <- scan("ssym_tensors_fwd_ddg.txt", what = numeric())

# Collect predictions from 10 files into a matrix
pred_list <- lapply(1:5, function(i) scan(paste0("Ssym_DA0rotTN_predictions_", i, ".txt"), what = numeric()))
pred_matrix <- do.call(cbind, pred_list)

# Average predictions across files
avg_pred <- rowMeans(pred_matrix)

# Compute RMSD
rmsd_val <- rmsd(avg_pred, target)

# Compute Pearson correlation
cor_val <- cor(target, avg_pred, method = "pearson")

# Print results
cat("RMSD:", rmsd_val, "\n")
cat("Pearson correlation:", cor_val, "\n")
