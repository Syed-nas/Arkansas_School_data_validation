#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np


# In[35]:


# Load the CSV file into a DataFrame
df = pd.read_csv("C:/Users/syednas2/Downloads/file 11 12 by year.csv")


# In[36]:


df.school.nunique()


# In[37]:


df


# In[38]:


li = list(df.iloc[0:,8:27].columns)
for i in li:
    df[i] = pd.to_numeric(df[i],errors='coerce')
    df[i] = df[i].fillna(0).astype('float')
df

    


# In[39]:


df.dtypes


# In[40]:


df.describe()


# In[41]:


df.info()


# In[42]:


df['solution'] = pd.to_numeric(df['solution'],errors='coerce')
df['solution'] = df['solution'].fillna(0).astype('int32')


# In[43]:


df['program'] = pd.to_numeric(df['program'],errors='coerce')
df['program'] = df['program'].fillna(0).astype('int32')


# In[44]:


df['COHORT_first'] = pd.to_datetime(df['COHORT_first'],errors='coerce')
df['COHORT_first'] = df['COHORT_first'].dt.year
df['COHORT_first'] = df['COHORT_first'].fillna(0).astype('int64')


# In[45]:


df['COHORT_first'].unique()


# In[46]:


df


# In[47]:


data_with_solutions = df.query('maxyears==3 and solution>=0 and COHORT_first!=0')


# In[48]:


data_with_solutions['school'].nunique()


# In[49]:


data_with_solutions.columns


# In[75]:


from scipy.stats import skew, kurtosis
from scipy.stats import norm
import matplotlib.pyplot as plt
mean_val = data_with_solutions.Attend.mean()
median_val = data_with_solutions.Attend.median()
mode_val = data_with_solutions.Attend.mode()
std_val = data_with_solutions.Attend.std()
skew = skew(data_with_solutions.Attend)
kurtosis = kurtosis(data_with_solutions.Attend)

x = np.linspace(data_with_solutions.Attend.min(),data_with_solutions.Attend.max(),1000)
pdf = norm.pdf(x, mean_val, std_val)
plt.hist(data_with_solutions.Attend, bins=30, edgecolor='black')
plt.plot(x, pdf, label='Distribution Curve')
plt.axvline(mean_val,color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
plt.axvline(median_val,color='green', linestyle='--', label=f'Median: {median_val:.2f}')
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print(f'{skew} skewness of data')
print(f'{kurtosis} kurtosis of data')
std_val


# In[50]:


data_with_solutions
fg = data_with_solutions.groupby('school')['attendFRL'].agg(['sum','mean','median','count'])


# In[51]:


fg


# In[52]:


data_with_solutions = data_with_solutions.sort_values(by =['school','year'],ascending= True)


# In[53]:


gh = data_with_solutions.query('year==2023 and percentecommath==0')


# In[54]:


data_with_solutions = data_with_solutions.drop(gh.index)


# In[55]:


data_with_solutions.query("school=='Bayyari Elementary School'")


# In[56]:


yb = data_with_solutions.groupby('school')['year'].max().reset_index()
yb


# In[57]:


yb = data_with_solutions.groupby('school')['COHORT_first'].max().reset_index()
yb


# In[58]:


data_with_solutions.school.nunique()


# In[59]:


baya = data_with_solutions.query("school=='Bayyari Elementary School'")


# In[27]:


baya


# In[46]:


"""test = baya.iloc[0:,8:27]
cohort_year = baya.COHORT_first.unique()
school = list(baya.school.unique())
for i in test.columns:
    ax = baya.plot(x='year', y=i, kind='line', marker='o')
    plt.axvline(x=cohort_year, color='red', linestyle='--', label='Cohort year')
    plt.axvline(x=cohort_year-1, color='purple', linestyle='--', label='Buffer year')
    plt.xticks(ticks=df['year'], labels=df['year'])
    ax.grid(True)
    plt.legend()
    plt.title(f'Plot for All Years{school}')
    plt.xlabel('Year')
    plt.ylabel(i)
    plt.show()"""


# In[60]:


baya = data_with_solutions.query("school=='Bayyari Elementary School'")
data_with_solutions


# In[61]:


a = data_with_solutions.query("school=='Gardner Stem Magnet School'")
b = data_with_solutions.query("school=='Wilbur D. Mills High School'")
c = data_with_solutions.query("school=='Ivory Intermediate School'")
data_with_solutions = data_with_solutions.drop(a.index)
data_with_solutions = data_with_solutions.drop(b.index)
data_with_solutions = data_with_solutions.drop(c.index)
data_with_solutions.school.nunique()


# In[49]:


"""schools = data_with_solutions.school.unique()
for j in range(0,len(schools)):
    yu = data_with_solutions[data_with_solutions.school==schools[j]]
    test = yu.iloc[0:,8:27]
    cohort_year = yu.COHORT_first.unique()
    school = yu.school.unique()
    for i in test.columns:
        ax = yu.plot(x='year', y=i, kind='line', marker='o')
        plt.axvline(x=cohort_year, color='red', linestyle='--', label='Cohort year')
        plt.axvline(x=cohort_year-1, color='purple', linestyle='--', label='Buffer year')
        plt.xticks(ticks=df['year'], labels=df['year'])
        ax.grid(True)
        plt.legend()
        plt.title(f'Plot for All Years{school}')
        plt.xlabel('Year')
        plt.ylabel(i)
        plt.show()"""


# In[52]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

df = data_with_solutions

df['PLCatWork'] = df['solution'] 

df['Post'] = (df['year'] >= df['COHORT_first']).astype(int)

df['PLCatWork_Post'] = df['PLCatWork'] * df['Post']

outcome_var = 'percentallgrademath'

did_model = smf.ols(formula=f'{outcome_var} ~ PLCatWork + Post + PLCatWork_Post', data=df).fit()

print("Difference-in-Differences Model Results:")
print(did_model.summary())


# In[50]:


data_with_solutions.count()


# In[51]:


baya.count()


# In[52]:


me = baya['percentallgrademath'].mean()
st = baya['percentallgrademath'].std()


# In[53]:


df.dtypes


# In[58]:


baya


# In[65]:


con= baya.query('solution==0')
treat = baya.query('solution==1')
mean_con = con.allgrademath.mean()
mean_treat = treat.allgrademath.mean()
con_math_score = con.allgrademath
treat_math_score = treat.allgrademath
prog =  mean_treat-mean_con
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(treat_math_score, con_math_score, equal_var=False)

print("T-statistic:", t_stat)
print("P-value:", p_value)


# In[62]:


data_with_solutions.to_csv("C:/Users/syednas2/Downloads/logis.csv")


# In[63]:


data_with_solutions = pd.read_csv("C:/Users/syednas2/Downloads/logis.csv")
data_with_solutions


# In[79]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = data_with_solutions

feature_columns = ['FRL','attendFRL','Attend','Blackpercent','Hisppercent','whitepercent','iep','gadratelow','gradreate','teacherex','novice','percentecommath','percentallgrademath','percenteconlea','percentallgradelea','econmath','allgrademath','econlea','allgradeslea'] 

X = df[feature_columns]
y = df['FRL'] 
treatment = df['solution']

X_train, X_test, y_train, y_test, treatment_train, treatment_test = train_test_split(X, y, treatment, test_size=0.2, random_state=30)

rf = RandomForestRegressor(n_estimators=100, random_state=30)
rf.fit(X_train, y_train)

df_control = df[df['solution'] == 0]
df_treatment = df[df['solution'] == 1]

X_control = df_control[feature_columns]
X_treatment = df_treatment[feature_columns]

y_pred_control = rf.predict(X_control)
y_pred_treatment = rf.predict(X_treatment)

ate = y_pred_treatment.mean() - y_pred_control.mean()

print(f"Estimated Average Treatment Effect (ATE): {ate}")


# In[ ]:


'attendFRL','Attend','Blackpercent','Hisppercent','whitepercent','iep','gadratelow','gradreate','teacherex','novice','percentecommath','percentallgrademath','percenteconlea',
'percentallgradelea',
*'econmath','allgrademath',----value added score for math
*'econlea','allgradeslea'----value added score ELA


# In[86]:


df.drop(columns=['Unnamed: 0'],inplace=True)


# In[93]:


c = df.corr()
c


# In[ ]:




