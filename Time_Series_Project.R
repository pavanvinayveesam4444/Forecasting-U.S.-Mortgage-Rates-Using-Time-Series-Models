# Loading required libraries
library(forecast)
library(zoo)
library(readxl)
# Setting working directory to where our Excel file is located
setwd("/Users/pavanvinayveesam/Documents/Time_Series")
# Reading the Excel file 
mortgage_rates.data <- read_excel("mortgage_rates.xlsx")

# Creating the time series object
mortgage_rates.ts <- ts(mortgage_rates.data$Value,
               start = c(1971, 4), end = c(2025, 4), freq = 12)
# View the time series
print(mortgage_rates.ts)
#Plotting the Time Series Data
plot(mortgage_rates.ts, main = "Mortgage Rates Time Series", ylab = "Rate", xlab = "Year")
# Plotting the autocorrelation function (ACF) for the mortgage rates time series
autocor <- Acf(mortgage_rates.ts, lag.max = 12, 
               main = "Autocorrelation of Mortgage Rates")
# STL decomposition of mortgage rates
mortgage.stl <- stl(mortgage_rates.ts, s.window = "periodic")
autoplot(mortgage.stl, main = "STL Decomposition of Mortgage Rates")
# Create data partitioning for training and validation 
# data sets.
length(mortgage_rates.ts)
nValid <- 130
nTrain <- length(mortgage_rates.ts) - nValid
nTrain
train.ts <- window(mortgage_rates.ts, start = c(1971, 4), end = c(2014,6))
train.ts
length(train.ts)
valid.ts <- window(mortgage_rates.ts, start = c(2014,7), 
                   end = c(2025,4))
valid.ts
length(valid.ts)
# Checking for random walk 
# Use Arima() function to fit AR(1) model.
# The ARIMA model of order = c(1,0,0) gives an AR(1) model.
mortgage.ar1<- Arima(mortgage_rates.ts, order = c(1,0,0))
summary(mortgage.ar1)
ar1 <- 0.9944
s.e. <- 0.0034
null_mean <- 1
alpha <- 0.05
z.stat <- (ar1-null_mean)/s.e.
z.stat
p.value <- pnorm(z.stat)
p.value
if (p.value<alpha) {
  "Reject null hypothesis"
} else {
  "Accept null hypothesis"
}
# Create differenced mortgage_rates.ts data using lag-1.
diff.mortgage <- diff(mortgage_rates.ts, lag = 1)
# Use Acf() function to identify autocorrelation for the model differencing. 
# and plot autocorrelation for different lags (up to maximum of 24).
Acf(diff.mortgage, lag.max = 24, 
    main = "Autocorrelation for First Differencing of Mortgage Rates")





##Naive and Seasonal Naive Forecasting
## Naive Forecast
naive.fst <- naive(train.ts, h = nValid)
naive.fst
# Accuracy of naive forecast
round(accuracy(naive.fst$mean, valid.ts), 3)
## Seasonal Naive Forecast
snaive.fst <- snaive(train.ts, h = nValid)
snaive.fst
# Accuracy of seasonal naive forecast
round(accuracy(snaive.fst$mean, valid.ts), 3)




## Trailing Moving Averages Forecasting
# Creating trailing moving average with window width of k = 4
# In rollmean(), using argument align = "right" to calculate a trailing MA.
ma.trailing_4 <- rollmean(train.ts, k = 4, align = "right")
## Creating forecast for the validation data for the window width
# of k = 4
ma.trail_4.pred <- forecast(ma.trailing_4, h = nValid, level = 0)
ma.trail_4.pred
# Using accuracy() function to identify common accuracy measures.
# Using  round() function to round accuracy measures to three decimal digits.
round(accuracy(ma.trail_4.pred$mean, valid.ts), 3)




## Two-level forecast for the validation period by combining the regression forecast and 
## Trailing MA forecast for residuals
# Fit a regression model with linear trend and seasonality for
# training partition. 
trend.seas <- tslm(train.ts ~ trend  + season)
summary(trend.seas)
# Creating regression forecast with linear trend and seasonality for 
# validation period.
trend.seas.pred <- forecast(trend.seas, h = nValid, level = 0)
trend.seas.pred
trend.seas.res <- trend.seas$residuals
trend.seas.res
# Apply trailing MA for residuals with window width k = 4. 
ma.trail.res <- rollmean(trend.seas.res, k = 4, align = "right")
ma.trail.res
# Create residuals forecast for validation period.
ma.trail.res.pred <- forecast(ma.trail.res, h = nValid, level = 0)
ma.trail.res.pred
# Develop two-level forecast for validation period by combining  
# regression forecast and trailing MA forecast for residuals.
fst.2level <- trend.seas.pred$mean + ma.trail.res.pred$mean
fst.2level
# Create a table for validation period: validation data, regression 
# forecast, trailing MA for residuals and total forecast.
valid.df <- data.frame(valid.ts, trend.seas.pred$mean, 
                       ma.trail.res.pred$mean, 
                       fst.2level)
names(valid.df) <- c("Mortgage_rates", "Regression.Fst", 
                     "MA.Residuals.Fst", "Combined.Fst")
valid.df
# Use accuracy() function to identify common accuracy measures.
# Use round() function to round accuracy measures to three decimal digits.
round(accuracy(trend.seas.pred$mean, valid.ts), 3)
round(accuracy(fst.2level, valid.ts), 3)




#Advanced Exponential Smoothing Forecasting
# Using ets() function with model = "ZZZ", i.e., automated selection of
# error, trend, and seasonality options.
hw.ZZZ <- ets(train.ts, model = "ZZZ")
hw.ZZZ 
# Using forecast() function to make predictions using this model with 
# validation period (nValid). 
# Show predictions in tabular format.
hw.ZZZ.pred <- forecast(hw.ZZZ, h = nValid, level = 0)
hw.ZZZ.pred
# Evaluating the  accuracy of ETS(ZZZ) model on validation data
round(accuracy(hw.ZZZ.pred$mean, valid.ts), 3)










#Two-level forecasting model (regression model with linear trend and seasonality and 
#AR(1) model for residuals) 
# Using tslm() function to create linear trend and seasonality model.
train.trend.season <- tslm(train.ts ~ trend  + season)
# See summary of linear trend and seasonality equation 
# and associated parameters.
summary(train.trend.season)
# Applying forecast() function to make predictions for ts with 
# trend and seasonality model in validation set.  
train.trend.season.pred <- forecast(train.trend.season, h = nValid, level = 0)
train.trend.season.pred
# Using Acf() function to identify autocorrealtion for the 
# model residuals (training set), and plotting autocorrelation for 
# different lags (up to maximum of 8).
Acf(train.trend.season.pred$residuals, lag.max = 8, 
    main = "Autocorrelation for Training Residuals")
# Using Arima() function to fit AR(1) model for training residuals. 
# The Arima model of order = c(1,0,0) gives an AR(1) model.
# Using summary() to identify parameters of AR(1) model. 
res.ar1 <- Arima(train.trend.season$residuals, order = c(1,0,0))
summary(res.ar1)
# Using forecast() function to make prediction of residuals in validation set.
res.ar1.pred <- forecast(res.ar1, h = nValid, level = 0)
res.ar1.pred
# Using Acf() function to identify autocorrealtion for the training 
# residual of residuals and plot autocorrelation for different 
# lags (up to maximum of 8).
Acf(res.ar1$residuals, lag.max = 8, 
    main ="Autocorrelation for Training Residuals of Residuals")
# Creating two-level modeling results, regression + AR(1) for validation period.
valid.two.level.pred <- train.trend.season.pred$mean + res.ar1.pred$mean
valid.two.level.pred
round(accuracy(valid.two.level.pred,valid.ts),3)






#ARIMA and Auto ARIMA Forecasting Models
# Using Arima() function to fit ARIMA(1,1,1)(1,1,1) model 
# for trend and seasonality.
# Using summary() to show ARIMA model and its parameters.
train.arima <- Arima(train.ts, order = c(1,1,1), seasonal = c(1,1,1))
summary(train.arima)
train.arima.pred <- forecast(train.arima, h = nValid, level = 0)
train.arima.pred
# Utilizing auto.arima() function to automatically identify 
# the ARIMA model structure and parameters. 
# Developing the ARIMA forecast for the validation period. 
train.auto.arima <- auto.arima(train.ts)
summary(train.auto.arima)
train.auto.arima.pred <- forecast(train.auto.arima, 
                                  h = nValid, level = 0)
train.auto.arima.pred
# Accuracy measures for the two ARIMA models
round(accuracy(train.arima.pred$mean, valid.ts), 3)
round(accuracy(train.auto.arima.pred$mean, valid.ts), 3)






#Accuracy of all the Models 
# Accuracy of Naive Forecast
round(accuracy(naive.fst$mean, valid.ts), 3)
# Accuracy of Seasonal Naive Forecast
round(accuracy(snaive.fst$mean, valid.ts), 3)
# Accuracy of Moving Average Forecast (Trailing MA with k = 4)
round(accuracy(ma.trail_4.pred$mean, valid.ts), 3)
# Accuracy of Regression Model (Linear Trend + Seasonality)
round(accuracy(trend.seas.pred$mean, valid.ts), 3)
# Accuracy of Two-Level Forecast: Regression + Trailing MA on Residuals
round(accuracy(fst.2level, valid.ts), 3)
# Accuracy of ETS (ZZZ) Model
round(accuracy(hw.ZZZ.pred$mean, valid.ts), 3)
# Accuracy of Two-Level Forecast: Regression + AR(1) on Residuals
round(accuracy(valid.two.level.pred, valid.ts), 3)
# Accuracy of Manual ARIMA(1,1,1)(1,1,1) Forecast
round(accuracy(train.arima.pred$mean, valid.ts), 3)
# Accuracy of Auto ARIMA Forecast
round(accuracy(train.auto.arima.pred2ed$mean, valid.ts), 3)









##Applying the best model identified through validation data to the entire dataset 
Full.auto.arima <- auto.arima(mortgage_rates.ts)

# Display model details
summary(Full.auto.arima)

# Generate in-sample predictions (fitted values for the entire dataset)
Full.auto.arima.fitted <- fitted(Full.auto.arima)
Full.auto.arima.fitted

# Calculate accuracy between fitted values and actual mortgage rates
accuracy_metrics <- round(accuracy(Full.auto.arima.fitted, mortgage_rates.ts), 3)

# Print accuracy metrics
print(accuracy_metrics)
