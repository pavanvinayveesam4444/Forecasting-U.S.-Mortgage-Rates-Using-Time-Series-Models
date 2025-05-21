# Forecasting-U.S.-Mortgage-Rates-Using-Time-Series-Models
# Forecasting U.S. Mortgage Rates Using Time Series Models

##  Project Overview

This project applies advanced Time Series Forecasting techniques to analyze and predict U.S. 30-year fixed mortgage rates using historical monthly data from 1971 to 2025. The goal is to develop accurate, interpretable models that capture trends and seasonal behavior in mortgage rates, aiding financial forecasting and economic planning.

Using R and the `forecast` package, we implement the 8-step forecasting framework to explore multiple models and select the most accurate one based on RMSE, MAPE, and residual diagnostics. The final model selected was **Auto ARIMA**, which demonstrated robust forecasting performance.

---

##  Why This Project Was Built

U.S. mortgage rates are key economic indicators, impacting homebuyers, lenders, and the broader economy. Accurate forecasting can:

- Assist homebuyers and real estate professionals in planning financing decisions
- Help financial institutions better manage mortgage-backed securities and lending risk
- Contribute to economic research and policymaking

This project was designed to:

- Forecast mortgage rate trends using historical data  
- Evaluate and compare statistical models for time series forecasting  
- Build an Auto ARIMA model and validate it with real-world performance metrics  

---

##  Use Cases

This model has practical applications in:

- **Real Estate Planning** – Helping homebuyers and advisors make informed financial decisions  
- **Banking & Lending** – Supporting interest rate strategy and loan pricing  
- **Government & Policy** – Guiding economic policy through forward-looking rate indicators  
- **Financial Forecasting** – Assisting hedge funds and mortgage analysts in risk management  
- **Academic Research** – Demonstrating reproducible, data-driven forecasting for time series courses  

---

##  Technologies Used

- **R** – For time series modeling, visualization, and diagnostics  
- **forecast** – For ARIMA and exponential smoothing model training and evaluation  
- **fpp3 / tseries** – For decomposition, ACF/PACF, and stationarity testing  
- **ggplot2** – For clear and elegant time series visualizations  
- **Tidyverse** – For data wrangling and transformation  
- **RStudio** – As the primary development environment  

---

##   How This Project Is Useful

This project showcases the practical application of the 8-step forecasting method, enabling:

- Replication of forecasting workflows using real-world financial data  
- Comparison of baseline and advanced models (Naïve, ETS, Auto ARIMA)  
- Insightful visualizations of time trends, seasonality, and forecast intervals  
- Error diagnostics to assess and validate time series assumptions  

The techniques demonstrated here can be reused across other economic indicators such as interest rates, inflation, or unemployment figures.

---

##  Model Highlights

- **Final Model**: Auto ARIMA (0,1,3)
- **Test RMSE**: 1.441
- **Test MAPE**: 21.97%
- **Residuals**: No significant autocorrelation, normally distributed errors


---

## 📄 References

- [Macrotrends: 30-Year Fixed Mortgage Rate Chart](https://www.macrotrends.net/2604/30-year-fixed-mortgage-rate-chart)  
- Hyndman & Athanasopoulos – *Forecasting: Principles and Practice*  
- R Documentation – `forecast`, `fpp3`, `tseries`
