import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from scipy import stats

#I added plotly later because I really despise matplot

#Starting general overview: I am going to load the data for S&P 500 index and Ferrari N.V. stock value

#----------------------------------------------

#Section 1: Data collection
#Here I just create the dataframes of the two companies
data = yf.download(["RACE", "^GSPC"], start = "2020-01-01", end = "2026-01-01")
#I printed the closing prices and plotted closing prices graph(lowkey forgetting that S&P 500 is obviously higher)
# data["Close"].plot(title = "close: Ferrari vs S&P500")
# plt.show()
#Here I printed various statistical indicators and checked if the dataframe was not flawed in some way
# print(data.describe())
# print(data)
# print(data.shape)
# print(data.columns)

#Here I print the first and last closing valeus:
print(data["Close", "RACE"].iloc[0])
print(data["Close", "RACE"].iloc[-1])
print(data["Close", "^GSPC"].iloc[0])
print(data["Close", "^GSPC"].iloc[-1])

#Seeing that these values look healthy I calculate the total return:
race_tr = (data["Close", "RACE"].iloc[-1] / data["Close", "RACE"].iloc[0]) - 1
print(race_tr)
#we see that over the last 5 years Ferrari stock had a retun of approximately 62.2%
sp500_tr = (data["Close", "^GSPC"].iloc[-1] / data["Close", "^GSPC"].iloc[0]) - 1
print(sp500_tr)
#We see that over the last 5 years the S&P500 had a total return of 69.4%

#Now i will calculate the annualized return over the last 5 yers:

race_ar = (1 + race_tr) ** (1 / 5) - 1
sp500_ar = (1 + sp500_tr) ** (1 / 5) - 1
print(race_ar)
print(sp500_ar)

#Here we notice the the annualized retuns for ferrari are 10.7% while the S&P 500 had a 11% annualized return
#Now I do the following 
#1) create a new column for cumulative return for the 2 Stocks
#2) plot the two cumulative returns and compare

data["CR", "RACE"] = data["Close"]["RACE"] / data["Close"]["RACE"].iloc[0] - 1
data["CR", "^GSPC"] = data["Close"]["^GSPC"] / data["Close"]["^GSPC"].iloc[0] - 1

#Overall analysis of the graph:


#From the graph we can see that initially the cumulative returns of ferrari and the S&P 500 were similar

#We can notice in the graph the ferrari's Cumulative return suddently plummets
#Obviously something must have happened in that period therefore I decided to research on the internet the reasons related to this drop.
#I found out that the drop dates to October 9th 2025, when ferrari held its Capital Markets day presentation.
#The stock plummeted down from a value of aprox $500 to $400. This is because investors were disappointed by the changing declared financial forecast.
#Specificallt investors were disappointed by the following:
# 1) the 5% target growth rate is a stepdown compared to the double digited growth we saw before
# 2) the company declared to scale back on EV ambition

#this is the plot made with plotly express
fig = px.line(data["CR"].reset_index(), x='Date', y=["RACE", "^GSPC"], title='Cumulative return: Ferrari vs S&P500')
fig.show()  # opens in browser, hover to see exact values


#SECTION 2: __________________________________________
#Here I calculate the daily excess return of ferrari stock and then comparing it to the S&P500 and then checking if there is a relationship between the two
#My first step Is calculating the risk free rate with a 13 week T-bill:
#Now I just gather the daily returns from the 2 different indeces
d_returns = data["Close"].pct_change().dropna()
d_returns["excess"] = d_returns["RACE"] - d_returns["^GSPC"]

fig0 = px.line(d_returns.reset_index(), x='Date', y="excess", title='Daily Excess Return: Ferrari vs S&P500')
fig0.show()
#Since the line is too chaotic to analyse I decided that I will take in account the following to think about performance:
# 1) the percentage of times that the line is above 
# 2) the cumulative returns we 
pct_above_zero = (d_returns["excess"] > 0).mean() * 100
print(f"Ferrari outperformed S&P500 on {pct_above_zero:.1f}% of days")
#Ferrari outperformed the S&P500 49.8% of the days
#However ferrari outperformes the S&P500 in cumulative returns
#From January 1st 2020 to January 1st 2026 ferrari had a total of 1.30 in cumulative returns (130%)
#In the same period of time the S&P500 had a cumulative return of 1.16 (116%)
#Therefore we can assess that ferrari outperformed the S&P 500 in this period.

#CORRELATION BETWEEN THE TWO:
correlation = d_returns["RACE"].corr(d_returns["^GSPC"])
print(f"Correlation between Ferrari and S&P500: {correlation:.4f}")
#The correlation between ferrari and S&P500 is 0.6329
#This means that more than half of the time the two stocks move together
#This makes sense as ferrari stocks are not immune to general sellofs (we can notice on the cumulative return graphs that there are massive sellofs both in ferrari and the S&P 500 dropped in 2022 and 2025)
#this can be linked to the brand's unique characteristics:
# 1) since it is a luxury brand, its customers is  immune to regular economic cycles
# 2) The stock is influenced by its internal news e.g. new car models / models price raises
# 3) Ferrari's Capital Markets day presentation (October 9th 2025)


#SECTION 3: LINEAR REGRESSION ________________________
x = d_returns["^GSPC"]
y = d_returns["RACE"]

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"Beta (slope): {slope:.4f}")
print(f"Alpha (intercept): {intercept:.6f}")
print(f"R-squared: {r_value**2:.4f}")

#LINEAR REGRESSION ANALYSIS:

#the slope - or beta - of the regression line is 0.9249, this indicates that ferrari moves roughly 0.92% for every 1% that the market moves
#since it is under a 1% value this means that Ferrari stock is less volative than S&P 500. 
#this is due to the fact that Ferrari's clients are generally not affected by market shifts (opposed clients of some S&P 500 companies)

#The y-intercept - or alpha - of the regression line is 0.000198. This means that 0.02% of Ferrari's daily profits are independent of general market trends
#Over the course of the year this small percentage ends up compounding to a net of 0.02 x 252 ~= 5% of extra annual returns

#The R sqaured value is 0.4006 which indicates that only 40% of ferrari's daily movements are dictated by the market, the other 60 percent is determined by specific ferrari factors
#From this value we can conclude that Ferrari behaves independently from the market


fig0 = px.line(d_returns.reset_index(), x='Date', y="excess", title='Daily Excess Return: Ferrari vs S&P500')
fig0.show()

plt.scatter(d_returns["^GSPC"], d_returns["RACE"], alpha=0.3)
plt.plot(d_returns["^GSPC"], slope * d_returns["^GSPC"] + intercept, color="red", label=f"y={slope:.2f}x+{intercept:.4f}")
plt.xlabel("S&P500 Daily Returns")
plt.ylabel("Ferrari Daily Returns")
plt.title("Ferrari vs S&P500 Returns Regression")
plt.legend()
plt.show()