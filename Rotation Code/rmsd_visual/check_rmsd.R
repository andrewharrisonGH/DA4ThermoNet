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

ggplot(data, aes(x = xrot_clock, y = yrot_clock, fill = rmsd)) +
  geom_tile() +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(title = "xrot vs yrot colored by RMSD",
       x = "X Rotation",
       y = "Y Rotation",
       fill = "RMSD")

library(akima)

interp_data <- with(data, interp(x = xrot_clock, y = yrot_clock, z = rmsd))
plot_ly(x = ~interp_data$x, y = ~interp_data$y, z = ~interp_data$z, type = "surface") %>%
  layout(
    title = "3D Interpolated RMSD Surface",
    scene = list(
      xaxis = list(title = "X Rotation"),
      yaxis = list(title = "Y Rotation"),
      zaxis = list(title = "RMSD")
    )
  )

# Load libraries
library(akima)
library(plotly)
library(readr)
library(dplyr)

# Read your data
data <- read_csv("rmsd_log.csv")  # Replace with actual file path

# Ensure (0, 0) is included in interpolation domain
if (!any(data$xrot_clock == 0 & data$yrot_clock == 0)) {
  data <- bind_rows(data, tibble(xrot_clock = 0, yrot_clock = 0, rmsd = mean(data$rmsd)))
}

# Interpolate onto a grid
interp_data <- with(data, interp(
  x = xrot_clock,
  y = yrot_clock,
  z = rmsd,
  xo = seq(0, 360, length.out = 100),  # full rotation grid
  yo = seq(0, 360, length.out = 100),
  duplicate = "mean"
))

# Plot as interactive 3D surface
plot_ly(
  x = interp_data$x,
  y = interp_data$y,
  z = interp_data$z,
  type = "surface",
  colorbar = list(title="RMSD"),
  colorscale = list(c(0, 'blue'), c(1, 'red'))
) %>%
  layout(
    title = "RMSD Surface of 1A23 (center=33) by X and Y Rotation (°)",
    scene = list(
      xaxis = list(title = "X Rotation (°)"),
      yaxis = list(title = "Y Rotation (°)"),
      zaxis = list(title = "RMSD", range = c(0, 40))
    )
  )
