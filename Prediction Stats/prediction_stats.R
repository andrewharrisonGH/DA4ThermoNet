# RMSD and PCC ------------------------------------------|
target_file <-"./S669/s669_tensors_fwd_ddg.txt"
pred_file <- "./S669/S669_i120TN_predictions_"

# Function to compute RMSD
rmsd <- function(pred, target) {
  sqrt(mean((pred - target)^2))
}

# Read target values
target <- scan(target_file, what = numeric())

# Collect predictions from 10 files into a matrix
pred_list <- lapply(1:10, function(i) scan(paste0(pred_file, i, ".txt"), what = numeric()))
pred_matrix <- do.call(cbind, pred_list)
#single file
#avg_pred <- scan("./S669/S669_ThermoNet_predictions_10.txt", what = numeric())

# Average predictions across files
avg_pred <- rowMeans(pred_matrix)
avg_pred <- -avg_pred

# Compute RMSD
rmsd_val <- rmsd(avg_pred, target)

# Compute Pearson correlation
cor_val <- cor(target, avg_pred, method = "pearson")

# Compute MAE
n <- length(target)
mae_val <- mae_manual <- sum(abs(target - avg_pred)) / n

# Print results
cat("RMSD:", rmsd_val, "\n")
cat("Pearson correlation:", cor_val, "\n")
cat("Mean Absolute Error:", mae_val, "\n")

# fwd+rev PCC and delta -----------------------------------------------|
fwd_file <- "./S669/S669_ThermoNet_predictions_"
rev_file <- "./S669/S669r_ThermoNet_predictions_"

# Collect predictions from 10 files into a matrix
fwd_list <- lapply(1:10, function(i) scan(paste0(fwd_file, i, ".txt"), what = numeric()))
fwd_matrix <- do.call(cbind, fwd_list)

rev_list <- lapply(1:10, function(i) scan(paste0(rev_file, i, ".txt"), what = numeric()))
rev_matrix <- do.call(cbind, rev_list)

# Average predictions across files
avg_fwd_pred <- rowMeans(fwd_matrix)
avg_rev_pred <- rowMeans(rev_matrix)

delta_pred <- avg_fwd_pred + avg_rev_pred
cor(avg_fwd_pred, avg_rev_pred)
mean(delta_pred)


# Getting 95% CI -----------------------------------------------------|
# Sample RMSD values
rmsd_values <- c(1.4, 1.5, 1.3, 1.6, 1.4, 1.2, 1.5, 1.3, 1.5, 1.4)

# Calculate mean and standard error
mean_rmsd <- mean(rmsd_values)
se_rmsd <- sd(rmsd_values) / sqrt(length(rmsd_values))

# Compute 95% confidence interval
ci_lower <- mean_rmsd - qt(0.975, df = length(rmsd_values) - 1) * se_rmsd
ci_upper <- mean_rmsd + qt(0.975, df = length(rmsd_values) - 1) * se_rmsd

# Output
cat("95% CI for RMSD:", ci_lower, "to", ci_upper, "\n")



# Make Boxplots for 10 Runs of 0 and i72 Augmentation ---------------|
# Function to compute RMSD
rmsd <- function(pred, target) {
  sqrt(mean((pred - target)^2))
}

# Read target values
target_file <-"./S669/s669_tensors_fwd_ddg.txt"
target <- scan(target_file, what = numeric())

# --- Function to collect RMSDs for one run ---
get_rmsds <- function(label) {
  sapply(1:10, function(e) {
    pred_list <- lapply(1:10, function(i) {
      scan(paste0("./S669/S669_", e, "_", label, "TN_predictions_", i, ".txt"), what = numeric())
    })
    
    pred_matrix <- do.call(cbind, pred_list)
    avg_pred <- -rowMeans(pred_matrix)
    
    rmsd(avg_pred, target)
  }) -> vals
  
  # return as data frame with label
  data.frame(label = label, rmsd = vals)
}

# --- Collect for both runs ---
results_0   <- get_rmsds("0")
results_i72 <- get_rmsds("i72")

all_results <- rbind(results_0, results_i72)

# --- Boxplot ---
boxplot(rmsd ~ label, data = all_results,
        main = "RMSD by Run",
        xlab = "Run",
        ylab = "RMSD")

#test Normality
shapiro.test(all_results$rmsd[all_results$label == "0"])
shapiro.test(all_results$rmsd[all_results$label == "i72"])

qqnorm(all_results$rmsd[all_results$label == "0"]); qqline(all_results$rmsd[all_results$label == "0"])
qqnorm(all_results$rmsd[all_results$label == "i72"]); qqline(all_results$rmsd[all_results$label == "i72"])

# If Normal:
t.test(rmsd ~ label, data = all_results)

#If Non-Normal:
wilcox.test(rmsd ~ label, data = all_results)

#--------------------------------------------------------------|


# Function to compute R2
rsq <- function(pred, target) {
  ss_res <- sum((target - pred)^2)
  ss_tot <- sum((target - mean(target))^2)
  1 - ss_res / ss_tot
}

r2_values <- sapply(1:10, function(e) {
  pred_list <- lapply(1:10, function(i) {
    scan(paste0("./S669/S669_", e, "_0TN_predictions_", i, ".txt"), what = numeric())
  })
  
  pred_matrix <- do.call(cbind, pred_list)
  avg_pred <- -rowMeans(pred_matrix)
  
  rsq(avg_pred, target)
})

print(mean(r2_values))



target_file <-"./S669/s669_tensors_rev_ddg.txt"
target <- scan(target_file, what = numeric())

# Compute R2 for both runs
get_r2 <- function(label) {
  sapply(1:10, function(e) {
    pred_list <- lapply(1:10, function(i) {
      scan(paste0("./S669/S669r_", e, "_", label, "TN_predictions_", i, ".txt"), what = numeric())
    })
    pred_matrix <- do.call(cbind, pred_list)
    avg_pred <- -rowMeans(pred_matrix)
    rsq(avg_pred, target)
  }) -> vals
  data.frame(label = label, r2 = vals)
}

r2_0   <- get_r2("0")
r2_i72 <- get_r2("i72")
all_r2_results <- rbind(r2_0, r2_i72)

# Parametric test
t.test(r2 ~ label, data = all_r2_results)

# Non-parametric test
wilcox.test(r2 ~ label, data = all_r2_results)

library(boot)

diff_mean <- function(data, indices) {
  d <- data[indices, ]
  mean(d$r2[d$label == "i72"]) - mean(d$r2[d$label == "0"])
}

boot_diff <- boot(all_r2_results, diff_mean, R = 10000)
boot.ci(boot_diff, type = "perc")


# --- Function to compute MAE ---
mae <- function(pred, target) {
  mean(abs(pred - target))
}

# --- Read target values ---
target_file <- "./S669/s669_tensors_rev_ddg.txt"
target <- scan(target_file, what = numeric())

# --- Function to collect MAEs for one run ---
get_maes <- function(label) {
  sapply(1:10, function(e) {
    pred_list <- lapply(1:10, function(i) {
      scan(paste0("./S669/S669r_", e, "_", label, "TN_predictions_", i, ".txt"), 
           what = numeric())
    })
    
    pred_matrix <- do.call(cbind, pred_list)
    avg_pred <- -rowMeans(pred_matrix)
    
    mae(avg_pred, target)
  }) -> vals
  
  # return as data frame with label
  data.frame(label = label, mae = vals)
}

# --- Collect for both runs ---
results_0   <- get_maes("0")
results_i72 <- get_maes("i72")

all_results <- rbind(results_0, results_i72)

# --- Boxplot ---
boxplot(mae ~ label, data = all_results,
        main = "MAE by Run",
        xlab = "Run",
        ylab = "MAE")

# --- Normality tests ---
shapiro.test(all_results$mae[all_results$label == "0"])
shapiro.test(all_results$mae[all_results$label == "i72"])

qqnorm(all_results$mae[all_results$label == "0"]); qqline(all_results$mae[all_results$label == "0"])
qqnorm(all_results$mae[all_results$label == "i72"]); qqline(all_results$mae[all_results$label == "i72"])

# --- If Normal:
t.test(mae ~ label, data = all_results)

# --- If Non-Normal:
wilcox.test(mae ~ label, data = all_results)



# --- Function to compute Pearson correlation ---
pearson_corr <- function(fwd_matrix, rev_matrix) {
  avg_fwd_pred <- rowMeans(fwd_matrix)
  avg_rev_pred <- rowMeans(rev_matrix)
  cor(avg_fwd_pred, avg_rev_pred, method = "pearson")
}

# --- Function to collect correlations for one run ---
get_corrs <- function(label) {
  sapply(1:10, function(e) {
    fwd_list <- lapply(1:10, function(i) {
      scan(paste0("./S669/S669_", e, "_", label, "TN_predictions_", i, ".txt"), 
           what = numeric())
    })
    rev_list <- lapply(1:10, function(i) {
      scan(paste0("./S669/S669r_", e, "_", label, "TN_predictions_", i, ".txt"), 
           what = numeric())
    })
    
    fwd_matrix <- do.call(cbind, fwd_list)
    rev_matrix <- do.call(cbind, rev_list)
    
    pearson_corr(fwd_matrix, rev_matrix)
  }) -> vals
  
  # return as data frame with label
  data.frame(label = label, corr = vals)
}

# --- Collect for both runs ---
results_0   <- get_corrs("0")
results_i72 <- get_corrs("i72")

all_results <- rbind(results_0, results_i72)

# --- Boxplot ---
boxplot(corr ~ label, data = all_results,
        main = "Pearson Correlation by Run",
        xlab = "Run",
        ylab = "Correlation (r)")

# --- Normality tests ---
shapiro.test(all_results$corr[all_results$label == "0"])
shapiro.test(all_results$corr[all_results$label == "i72"])

qqnorm(all_results$corr[all_results$label == "0"]); qqline(all_results$corr[all_results$label == "0"])
qqnorm(all_results$corr[all_results$label == "i72"]); qqline(all_results$corr[all_results$label == "i72"])

# --- If Normal:
t.test(corr ~ label, data = all_results)

# --- If Non-Normal:
wilcox.test(corr ~ label, data = all_results)