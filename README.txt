Code for the paper "Random forest for high-dimensional non-linear forecasting"

First reset working directories in the R files. Then run all of DataRead.R and all of DataAnalysis.R

You can reproduce results for RMSEs by running the following code:
mse(s,k,method)
where s is the forecast horizon (1 to 5), k the variable index (1 to 143) and method can be foress, dfs, svmfuns etc.
