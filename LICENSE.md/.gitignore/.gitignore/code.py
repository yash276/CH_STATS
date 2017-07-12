#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA

#reading the data
#modifying it a bit
#changing default index to date
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
df = pd.read_csv('oct_march.csv', parse_dates=True, index_col='date',date_parser=dateparse)

#creating the prediction dataframe
pred=pd.DataFrame(columns=[['views','subscriber','videoscount']])


#a function defined that fit ARIMA model of the order on the column 
# returns the predictions for the nnext three months
def merge (column,order):
    model = ARIMA(ts_log[column], order=order)  
    column = model.fit(disp=-1) 
    pred = column.predict(start='2017-04-01',end='2017-06-30')
    return pred
    
    

#actual algorithm for calculating the predictions for all 'chid' in the original data 

for chid in df['chid'].unique():
    #creating a dataframe for single chid
    ts=df[df['chid']==chid][['views','subscriber','videoscount']]
    #creating dataframe for final storage in csv file
    future_three=pd.DataFrame(columns=[['views','subscriber','videoscount']])
    #dropping of any columns that are irrelevant
    for column in ts.columns:
        if (ts[ts[column]==0][column].count()==len(ts)):
            ts.drop(labels=column,inplace=True,axis=1)
    #taking log to punish higher values or for making time series stationary
    ts_log=np.log(ts)
    #dropping the columns in ts_log with infinite value
    for column in ts_log.columns:
        if (ts_log[ts_log[column]==-np.inf][column].count()==len(ts_log)):
            ts_log.replace([np.inf, -np.inf], np.nan,inplace=True)
            ts_log.dropna(inplace=True,axis=1)
    #making the time series more smoother the taking a moving difference
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)
   #dropping any columns in ts_log that are not relevant in ts_log_diff
    for column in ts_log_diff.columns:
        if (ts_log_diff[ts_log_diff[column]==0.0][column].count()==len(ts_log_diff)):
            ts_log.drop(labels=column,inplace=True,axis=1)
    #fitting the ARIMA Time Series model for various 'chid' 
    #general is (4,1,2)
    #but some require different model for the SVD to converge
    for column in ts_log.columns:
        if(chid=='UC6BVe3qF72Npr6hBoQ'):
            order=(2,1,2)
            pred[column]=merge(column,order)
        elif(chid=='UC6YPrSElj_8A39DIvg' or chid=='UC6M1fGj_FokrrgqRuw' or chid=='UC68SaYcDGi6vrS4qGQ'or chid=='UC6R1DFp6TbjLRaBDNA'or chid=='UC6flofJskgS3FUpTNQ' or chid=='UC6BzZnZQhIZia8rKfw' or chid=='UC6IsogHvcPITLjkqSQ'or chid=='UC6tYn2JQgIsZ2ab2bw' or chid=='UC6SjJYxfFL1_vsHPdw' or chid=='UC6zwaRF75y2bghpaEA'):
            order=(2,1,1)
            pred[column]=merge(column,order)
        elif(chid=='UC685tbaDLsOlfBVzfA'or chid=='UC6lg1TT-CG93KH3ZWg' or chid=='UC6iYzKmOJO0Cs-WfCw' or chid=='UC66MdV183KKnHdvIiA' or chid=='UC6AbGQJtsLAjcGNs9w' or chid=='UC6JTYOqAl0rX4UE0zA'):
            order=(4,1,1)
            pred[column]=merge(column,order)
       #I was not able to predict the values of these 'chid' due to some errors encountered
       #so for now let the loop run and skipped over to other values
        elif(chid=='UC6SUIeJ5frJm3A4FJg' or chid=='UC6Ds9AtRgfoGPOr6-A'or chid=='UC68mjRivBWSApWXTog'or chid=='UC6zmKdHGKMTBxHG6Hg' or chid=='UC6jRSvabqGqw7FwikA' or chid=='UC6PFoeUC65WJg' or chid=='UC6Bv0PiBObuYcBzF2Q' or chid=='UC6iyxTNq8EzJ0qHirw'):
            continue 
        else:
            order=(4,1,2)
            pred[column]=merge(column,order)
    #converting the predicted data back to the normal form
    #introducing back trend and stationarity properties
    for column in ts_log.columns:
        future_cumsum = pred.cumsum()
        future_log = pd.Series(ts_log[column][len(ts_log)-1],index=future_cumsum.index)
        future_log = future_log.add(future_cumsum[column],fill_value=0)
        future_final=np.exp(future_log)
        future_three[column]=future_final
    
    #fixing the final data to be stored
    future_three.reset_index(inplace=True)
    future_three.columns=['date', 'views', 'subscriber', 'videoscount']
    future_three['chid']=chid
    for column in ts.columns:
        if(future_three[future_three[column]!=None][column].count()==0):
            future_three.drop(labels=column,inplace=True,axis=1)
            future_three[column]=ts[column][0]
    future_three.fillna(value=0)
    # finally writing the csv file
    future_three.to_csv('april_june.csv',index=False,mode='a+')


