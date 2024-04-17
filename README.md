# UChicago Trading Competition Code
This repository has Princeton's code for the 2024 UChicago Trading Competition, which earned 2nd place for case 1, top 10 (exact place not known) for case 2, 
and 2nd place for the overall competition (out of 40 teams). The team consisted of Sabrina Van, Jerry Han, Arjun Menon, and Emily Gai. 

## Case 1
This case focused on market making given 5 stocks, 2 ETFs, and a risk free asset. The simulation was a 3-hour live trading session of 
5-minute intervals that increase in difficulty against
other competitors. Our primary strategy was to take advantage of the ETF arbitrage. While this was successful in beginning rounds, 
we switched to a market making strategy on the risk-free asset as ETF arbitrage became more minimial. 


## Case 2
The case focused on portoflio optimization that maximized Sharpe Ratio. Given historical data, we were to output a daily set of 
weights for how we wante dto allocated our portfolio. We took a direct approach: at each time step, we looked at the covariance 
between the stocks over the past 40 days and found the weights that maximized the Sharpe Ratio. We further normalized it 
to make sure it met weight constraints. Other files in the case2 folder includes other strategies that we attempted and used 
as benchmarks, however, 'finalCase2Code.py' was the submitted code. 
