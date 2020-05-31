# -*- coding: utf-8 -*-
"""
Created on Mon May 25 11:25:33 2020

@author: Administrator
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pylab as pl

def get_special_event(calendar):
    # get evernt type variable
    x = calendar[['event_name_1', 'event_type_1']].drop_duplicates()
    y = calendar[['event_name_2', 'event_type_2']].drop_duplicates()
    x = x.rename(columns={'event_name_1':'event_name','event_type_1':'event_type'})
    y = y.rename(columns={'event_name_2':'event_name','event_type_2':'event_type'})
    # event is a sheet listed all the special event name and event type
    event = x.append(y) 
    return event

def plot_price(prices):
   mean_prices_wk = prices.groupby(['wm_yr_wk', 'store_id']).mean().reset_index()
   sns.scatterplot(x='wm_yr_wk', y='sell_price', data=mean_prices_wk, size=1)  


def plot_sales_most_least(sales):
    # we check the sold number of this particular product
    sales['total'] = sales.iloc[:,6:].sum(axis=1)
    sales_sum = sales[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'total']]   
    most_sold_products = sales_sum[['item_id', 'total']].groupby('item_id').sum().nlargest(10, 'total').reset_index()
    least_sold_products = sales_sum[['item_id', 'total']].groupby('item_id').sum().nsmallest(10, 'total').reset_index()  
    plt.bar(x=most_sold_products['item_id'],height=most_sold_products['total'],width=0.5,color=['r','b'])
    pl.xticks(rotation=90)
    plt.ylabel('sold number')
    plt.xlabel('item name')
    plt.title('most sold products')
    plt.show()    
    plt.bar(x=least_sold_products['item_id'],height=least_sold_products['total'],width=0.5,color=['r','b'])
    pl.xticks(rotation=90)
    plt.ylabel('sold number')
    plt.xlabel('item name')
    plt.title('least sold products')
    plt.show()
    
    
def sales_by_department(sales):
    sns.set()
    sales['total'] = sales.iloc[:,6:].sum(axis=1)
    sales_sum = sales[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'total']]
    sales_by_department = sales_sum[['dept_id', 'total']].groupby('dept_id').sum().sort_values('total', ascending=False).reset_index()   
    plt.bar(x=sales_by_department['dept_id'],height=sales_by_department['total'],width=0.5,color=['r','b'])
    pl.xticks(rotation=90)
    plt.ylabel('sold number')
    plt.xlabel('department')
    plt.title('total sold number of each department')   
    plt.show()

def sales_by_store(sales):
    sns.set()
    sales['total'] = sales.iloc[:,6:].sum(axis=1)
    sales_sum = sales[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'total']]
    sales_by_store = sales_sum[['store_id', 'total']].groupby('store_id').sum().sort_values('total', ascending=False).reset_index()
    plt.bar(x=sales_by_store['store_id'],height=sales_by_store['total'],width=0.5,color=['r','b'])
    pl.xticks(rotation=90)
    plt.ylabel('sold number')
    plt.xlabel('store')
    plt.title('total sold number of each store')   
    plt.show()


    
def sales_by_state(sales):
    sns.set()
    sales['total'] = sales.iloc[:,6:].sum(axis=1)
    sales_sum = sales[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'total']]
    sales_by_state = sales_sum[['state_id', 'total']].groupby('state_id').sum().sort_values('total', ascending=False).reset_index()    
    plt.bar(x=sales_by_state['state_id'],height=sales_by_state['total'],width=0.5,color=['r','g','b'])
    pl.xticks(rotation=90)
    plt.ylabel('sold number')
    plt.xlabel('state')
    plt.title('total sold number of each state')  
    plt.show()

def sales_by_state_avg(sales):
    sns.set()
    sales['total'] = sales.iloc[:,6:].sum(axis=1)
    sales_sum = sales[['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'total']]
    sales_by_state_avg = sales_sum[['state_id', 'store_id', 'total']].groupby(['state_id', 'store_id']).sum().reset_index()
    sales_by_state_avg = sales_by_state_avg.groupby('state_id').mean().sort_values('total', ascending=False).reset_index()  
    plt.bar(x=sales_by_state_avg['state_id'],height=sales_by_state_avg['total'],width=0.5,color=['r','g','b'])
    pl.xticks(rotation=90)
    plt.ylabel('sold number')
    plt.xlabel('state')
    plt.title('average sold number of each state-store')  
    plt.show()
    
            
# data input

calendar = pd.read_csv('./data/calendar.csv')
prices = pd.read_csv('./data/sell_prices.csv')
sales = pd.read_csv('./data/sales_train_validation.csv')

plot_price(prices)
plot_sales_most_least(sales)
sales_by_department(sales)
sales_by_store(sales)
sales_by_state(sales)
sales_by_state_avg(sales)















