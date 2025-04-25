# Load libraries
# install.packages("xts")
library(xts)
library(zoo)

# Read CSV (skip first 2 rows)
san <- read.csv("dane/san.csv", skip = 2, row.names = 1)

# Convert row names (dates) to Date type
dates <- as.Date(rownames(san), format = "%Y-%m-%d")

# Create xts object for time series handling
san_xts <- xts(san[,1], order.by = dates)

# Compute log returns
log_return <- diff(log(1 + lag(san_xts, k = -1)))

# Remove NA values
log_return <- na.omit(log_return)

# Apply the transformation: 100 * (1 - exp(log_return))
ls <- 100 * (1 - exp(log_return))

# Split into train (80%) and test (20%)
n <- nrow(ls)
train <- ls[1:floor(0.8 * n)]
test <- ls[(floor(0.8 * n) + 1):n]

# Plot training data
plot(train, main = "Train Set", col = "blue", major.ticks = "years", ylab = "Transformed Return")

# install.packages("rugarch")
library(rugarch)


# Define the model specification
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(1, 1), include.mean = TRUE),
  distribution.model = "std"  # You can also use "std", "ged", etc.
)

fit <- ugarchfit(spec = spec, data = train)
# Print the fit summary
print(fit)
# roll one step ahead forecast for the test set
roll <- ugarchroll(spec,
                   ls,
                   n.ahead = 1,
                   forecast.length = length(test),
                   refit.every = 1,
                   refit.window = "moving",
                   solver = "hybrid",
                   calculate.VaR = TRUE,
                   VaR.alpha = c(0.95, 0.99))
# plot the forecast against the test set
roll
# extract VaR
VaR <- roll@forecast$VaR
# plot daily VaR
library(ggplot2)
library(dplyr)
ggplot(data.frame(Date = index(test), VaR = VaR), aes(x = Date, y = VaR)) +
  geom_line(color = "blue") +
  labs(title = "Daily VaR", x = "Date", y = "VaR") +
  theme_minimal()

VaRplot(0.9, VaR[3], VaR[2])
VaRTest(0.99, VaR[3], VaR[2])

sim <- ugarchsim(fit, n.sim = length(test), m.sim = 10000)
simulated_losses <- as.matrix(sim@simulation$seriesSim)

VaR_95 <- apply(simulated_losses, 1, quantile, probs = 0.95)
VaR_99 <- apply(simulated_losses, 1, quantile, probs = 0.99)

# install.packages("expectreg")
library(expectreg)
EVaR_95 <- apply(simulated_losses, 1, function(x) expectreg::expectile(x, 0.95))
EVaR_99 <- apply(simulated_losses, 1, function(x) expectreg::expectile(x, 0.99))

# Plot the VaR against the test set
ax1 <- ggplot(data.frame(Date = index(test), VaR_95 = VaR_95, VaR_99 = VaR_99), aes(x = Date)) +
  geom_line(aes(y = VaR_95), color = "blue") +
  geom_line(aes(y = VaR_99), color = "red") +
  geom_line(aes(y = test), color = "green") +
  labs(title = "Simulated VaR (95% and 99%)", x = "Date", y = "Simulated VaR") +
  theme_minimal()
ax2 <- ggplot(data.frame(Date = index(test), EVaR_95 = EVaR_95, EVaR_99 = EVaR_99), aes(x = Date)) +
  geom_line(aes(y = EVaR_95), color = "blue") +
  geom_line(aes(y = EVaR_99), color = "red") +
  geom_line(aes(y = test), color = "green") +
  labs(title = "Simulated EVaR (95% and 99%)", x = "Date", y = "Simulated EVaR") +
  theme_minimal()


library(patchwork)
ax1 / ax2
