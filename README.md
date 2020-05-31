# 6010U-final
Python code for Kaggle Contest m5-forecasting

## 1. Our Files
Our project includes two parts: the first part is EDA, which is in data_analysis.py, and the second part is model building, which is in xgboost-m5.py. The needed data are put in data file.To successfully run the code, it's necessary to set python working directory to be the unpacked folder (import os; os.setcwd(...)).

## 2. Main Idea of Our Codes
In data_analysis.py, we mainly analyse the sale_train validation.csv. We give some brief analysis of sale data. We check the sale number across differnet states, departments and stores and then plot the them out.

In xgboost-m5.py, we use calendar/item/price/sales information to generate numeric features (0,1,2,...) as input, and 28-day forward daily sales amount as output. the formatted data are stored in data.pkl. The predicted values are stored in submission.csv.

## 3. MemoryError to Notice
It should be noticed that model builing part need a  computer with large memory, otherwise the programme may crash (Memory Error). Therefore we select **5000** in 30490 time series to make 28-day ahead prediciton.
