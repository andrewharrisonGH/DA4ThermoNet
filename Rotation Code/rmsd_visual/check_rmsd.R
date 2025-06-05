library(ggplot2)
data <- read.csv("rmsd_log.csv")
plot(data$xrot_clock, data$rmsd)

ggplot(data, aes(x = xrot_clock, y = yrot_clock, color = rmsd)) +
  geom_point(size = 3) +
  scale_color_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(title = "xrot vs yrot colored by RMSD",
       x = "X Rotation",
       y = "Y Rotation",
       color = "RMSD")
