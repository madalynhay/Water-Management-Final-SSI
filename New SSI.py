#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# SSI Calculated similarly to SPEI, replace precipitation data with streamflow data.


# In[26]:


import scipy


# In[1]:


import numpy as np
from scipy import stats
from scipy.stats import kstest
 import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[3]:


import pandas as pd

from standard_precip.spi import SPI
from standard_precip.utils import plot_index


# In[2]:


pip install standard-precip


# In[4]:


df1 = pd.read_csv ('/Users/madalynhay/Downloads/waterdata.csv')
print(df1)


# In[5]:


df1 = df1.rename(columns={"agency_cd\tsite_no\tdatetime\t100139_00060_00003\t100139_00060_00003_cd": "agency_cd\tsite_no\tdatetime\tflow"})


# In[6]:


df1['split'] = df1['agency_cd\tsite_no\tdatetime\tflow'].apply(lambda x: x.split('\t'))
df1['len_split'] = df1['split'].apply(lambda x: len(x))


# In[7]:


df1 = df1[df1.len_split == 5] #filter out only columns with all five attributes


# In[8]:


df1['Agency'] = df1['split'].apply(lambda x: x[0])
df1['Site_no'] = df1['split'].apply(lambda x: x[1])
df1['Datetime'] = df1['split'].apply(lambda x: x[2])
df1['Flow'] = df1['split'].apply(lambda x: x[3])
df1['Approved'] = df1['split'].apply(lambda x: x[4])


# In[9]:


# Now 'df' is a pandas DataFrame containing your CSV data
df1.head()


# In[10]:


df1 = df1[df1.Flow.str.isnumeric()] #filter out all non-numeric flow entries


# In[11]:


#df1.Flow.astype(int) #convert int

df1['Flow'] = df1['Flow'].astype('int')


# In[12]:


# remove special character
df1.columns = df1.columns.str.replace(' ', '')


# In[13]:


df1.head()


# In[14]:


column_to_check = 'Flow'
column_to_check_str = str(column_to_check)

if column_to_check_str in df1.columns:
    values = df1[column_to_check_str]
    print(values)
else:
    print(f"{column_to_check} not found in the DataFrame.")


# In[15]:


spi = SPI()


# In[45]:


df1_spi = spi.calculate(
    df1, 
    'Datetime', 
    'Flow', 
    freq="D", 
    scale=1, 
    fit_type="mle", 
    dist_type="nor"
)


# In[129]:


df1_spi_nona = df1_spi.dropna()


# In[131]:


df1_spi_nona.head()


# In[132]:


fig = plot_index(df1_spi_nona, 'Datetime', 'Flow_calculated_index')
plt.title("SSI Normal Distribution")


# In[133]:


fig, ax = plt.subplots()
sm.qqplot(df1_spi_nona.Flow_calculated_index, line='45', ax=ax)
ax.set_title('Q-Q Plot Normal')
plt.show()


# In[84]:


df1_spi_gev = spi.calculate(
    df1, 
    'Datetime', 
    'Flow', 
    freq="D", 
    scale=1, 
    fit_type="mle", 
    dist_type="gev"
)


# In[137]:


df1_spi_gev_nona = df1_spi_gev.dropna()


# In[152]:


total_Flow_calculated_index = len(df1_spi_gev.Flow_calculated_index)
print(total_Flow_calculated_index)


# In[158]:


df1_spi_gev_Lessthan0 = df1_spi_gev.Flow_calculated_index[df1_spi_gev.Flow_calculated_index < 0.0 ].count() 
print(df1_spi_gev_Lessthan0)

df1_spi_gev_Lessthan2 = df1_spi_gev.Flow_calculated_index[df1_spi_gev.Flow_calculated_index < -2.0 ].count()
print(df1_spi_gev_Lessthan2)


# In[160]:


Flow_below0_P = (df1_spi_gev_Lessthan0 / total_Flow_calculated_index)
print(Flow_below0_P)

Flow_below2_P = (df1_spi_gev_Lessthan2 / total_Flow_calculated_index)
print(Flow_below2_P)


# In[164]:


df1_spi_gev_reverse = df1_spi_gev.sort_values( by=['Flow_calculated_index'], ascending = True)
print(df1_spi_gev_reverse)
print("The driest day is 1947-01-18.")


# In[167]:


rq0 = (1+ Flow_below0_P)
print(rq0)

rq2 = (1+ Flow_below2_P)
print(rq2)


# In[111]:


fig = plot_index(df1_spi_gev, 'Datetime', 'Flow_calculated_index')
plt.title("SSI Generalized Extreme Value Distribution")


# In[121]:


fig, ax = plt.subplots()
sm.qqplot(df1_spi_gev.Flow_calculated_index, line='45', ax=ax)
ax.set_title('Q-Q Plot Generalized Extreme Value')
plt.show()


# In[95]:


df1_spi_wei = spi.calculate(
    df1, 
    'Datetime', 
    'Flow', 
    freq="D", 
    scale=1, 
    fit_type="mle", 
    dist_type="wei"
)
df1_spi_wei.head()


# In[112]:


fig = plot_index(df1_spi_wei, 'Datetime', 'Flow_calculated_index')
plt.title("SSI Weibull Distribution")


# In[122]:


fig, ax = plt.subplots()
sm.qqplot(df1_spi_wei.Flow_calculated_index, line='45', ax=ax)
ax.set_title('Q-Q Plot Weibull')
plt.show()


# In[103]:


df1_spi_pe3 = spi.calculate(
    df1, 
    'Datetime', 
    'Flow', 
    freq="D", 
    scale=1, 
    fit_type="lmom", 
    dist_type="pe3"
)


# In[113]:


fig = plot_index(df1_spi_pe3, 'Datetime', 'Flow_calculated_index')
plt.title("SSI Pearson III Distribution")


# In[123]:


fig, ax = plt.subplots()
sm.qqplot(df1_spi_pe3.Flow_calculated_index, line='45', ax=ax)
ax.set_title('Q-Q Plot Pearson III')
plt.show()


# In[ ]:




