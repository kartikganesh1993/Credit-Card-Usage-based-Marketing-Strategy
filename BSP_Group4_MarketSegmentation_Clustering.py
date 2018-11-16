#Group 4 : Bopin Valsan, Latika CHopra, Isha Baiwar, kartik Ganesh
#Market Segmaentation using clustering

from sklearn.preprocessing import  StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score,silhouette_score
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#%matplotlib inline
import seaborn as sns
# reading data into dataframe
data= pd.read_csv("CC GENERAL.csv")
print "Head of Data is :"
print data.head()
print "Information of dataset is:"
print data.info()
print "Shape of data is:"
print data.shape
print "Data description is as follows :"
print data.describe()
print data['CREDIT_LIMIT'].isnull().value_counts()
print data['CREDIT_LIMIT'].describe()
print data[data['CREDIT_LIMIT'].isnull()]


## There are missing values in the data so we are replacing or imputing them with medians

print "NULL Data under each column is"
print data.isnull().sum()
data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].median(),inplace=True)
data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].median(),inplace=True)
print "After replacing all null values with medians column info of null values is"
print "Column wise NULL data after replacing with medians is:"
print data.isnull().sum()

## Defining new metrics
## METRIC 1 : Monthly_avg_purchase and Cash Advance Amount

data['Monthly_avg_purchase']=data['PURCHASES']/data['TENURE']
data['Monthly_cash_advance']=data['CASH_ADVANCE']/data['TENURE']
print data['Monthly_avg_purchase'].head()
print data['Monthly_cash_advance'].head()

print "number of One off purchases being zero is"
print data[data['ONEOFF_PURCHASES']==0]['ONEOFF_PURCHASES'].count()

## METRIC 2 : Purchase_type
#Data exploration to find what type of purchases customers are making on credit card.

print "When one off purchases is zero and installment purchases is also zero"
print data[(data['ONEOFF_PURCHASES']==0) & (data['INSTALLMENTS_PURCHASES']==0)].shape
print "When one off purchases is greater than zero and installment purchases is also greater than zero"
print data[(data['ONEOFF_PURCHASES']>0) & (data['INSTALLMENTS_PURCHASES']>0)].shape
print "When one off purchases is greater than zero and installment purchases is zero"
print data[(data['ONEOFF_PURCHASES']>0) & (data['INSTALLMENTS_PURCHASES']==0)].shape
print "When one off purchases is zero and installment purchases is greater than zero"
print data[(data['ONEOFF_PURCHASES']==0) & (data['INSTALLMENTS_PURCHASES']>0)].shape

# categorical variable being derived based on the behaviour.

def purchase(data):
    if (data['ONEOFF_PURCHASES']==0) & (data['INSTALLMENTS_PURCHASES']==0):
        return 'none'
    if (data['ONEOFF_PURCHASES']>0) & (data['INSTALLMENTS_PURCHASES']>0):
         return 'both_oneoff_installment'
    if (data['ONEOFF_PURCHASES']>0) & (data['INSTALLMENTS_PURCHASES']==0):
        return 'one_off'
    if (data['ONEOFF_PURCHASES']==0) & (data['INSTALLMENTS_PURCHASES']>0):
        return 'installment'
data['purchase_type']=data.apply(purchase,axis=1)
print "Count of distribution into categories is as follows :"
print data['purchase_type'].value_counts()

## METRIC 3 : Limit_Usage : (balance/credit limit)
#  Lower value means good credit score and good maintenance of credit score

data['limit_usage']=data.apply(lambda x: x['BALANCE']/x['CREDIT_LIMIT'], axis=1)
print data['limit_usage'].head()

## METRIC 4 : Payment to minimum payments ratio
data['payment_minpay']=data.apply(lambda x:x['PAYMENTS']/x['MINIMUM_PAYMENTS'],axis=1)

print "The data now stands at"
print data.info()
print "Payment to min payment ratios are described below"
print data['payment_minpay'].describe()

## Taking log of the data to remove the effect of outliers
print " Dropping CUST_ID and purchase type wjile doing the log transformation"

cr_log=data.drop(['CUST_ID','purchase_type'],axis=1).applymap(lambda x: np.log(x+1))
print cr_log.describe()

col=['BALANCE','PURCHASES','CASH_ADVANCE','TENURE','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','CREDIT_LIMIT']
cr_pre=cr_log[[x for x in cr_log.columns if x not in col ]]

# Average payment_minpayment ratio for each purchse type.
x=data.groupby('purchase_type').apply(lambda x: np.mean(x['payment_minpay']))
print type(x)
print x.values

print("Plotting ..... ")
fig,ax=plt.subplots()
ax.barh(bottom=range(len(x)),width=x.values)
ax.set(yticks= np.arange(len(x)),yticklabels=x.index);
plt.title('Mean payment_minpayment ratio for each purchse type')
plt.savefig("1_Mean payment_minpayment ratio for each purchse type")

print "Data Description now is"
print data.describe()

print data[data['purchase_type']=='n']
print("Plotting......")

data.groupby('purchase_type').apply(lambda x: np.mean(x['Monthly_cash_advance'])).plot.barh()
plt.plot()
plt.title('Average cash advance taken by customers of different Purchase type : Both, None,Installment,One_Off')
plt.savefig("2_Average cash advance taken by customers of different Purchase type")
 
data.groupby('purchase_type').apply(lambda x: np.mean(x['limit_usage'])).plot.barh()
plt.title('Understanding Credit Score ')
plt.savefig("3_Unerstanding_Credit_Score")

cre_original=pd.concat([data,pd.get_dummies(data['purchase_type'])],axis=1)
### APPLYING ML

# creating Dummies for categorical variable
cr_pre['purchase_type']=data.loc[:,'purchase_type']
print pd.get_dummies(cr_pre['purchase_type']).head()
cr_dummy=pd.concat([cr_pre,pd.get_dummies(cr_pre['purchase_type'])],axis=1)
l=['purchase_type']
cr_dummy=cr_dummy.drop(l,axis=1)
print "Table below is to check for null values"
print cr_dummy.isnull().sum()
print "Description"
print cr_dummy.describe()

print "Plotting the heatmap"
plot=sns.heatmap(cr_dummy.corr())
plt.savefig("4_Heatmap")

#Standardrizing data: This is done mainly to put data on the same scale
print("Using StandardScaler to standardize the data")
sc=StandardScaler()
cr_scaled=sc.fit_transform(cr_dummy)

# Applying Principal Component Analysis.. This will help us finally determine how many components to pick .
print "Applying PCA to undersatand how many components to pick"
var_ratio={}
for n in range(4,15):
    pc=PCA(n_components=n)
    cr_pca=pc.fit(cr_scaled)
    var_ratio[n]=sum(cr_pca.explained_variance_ratio_)
pc=PCA(n_components=5)
p=pc.fit(cr_scaled)
print cr_scaled.shape
print p.explained_variance_
print "Var Ratio across clusters"
print var_ratio

pd.Series(var_ratio).plot()
plt.title("Var_Ratio_Plot")
plt.savefig("5_VAR_Ratio_Plot")

print "We can go ahead with 5 Components as they explain close to 88% of the variance"

pc_final=PCA(n_components=5).fit(cr_scaled)

reduced_cr=pc_final.fit_transform(cr_scaled)
dd=pd.DataFrame(reduced_cr)
print dd.shape
col_list=cr_dummy.columns
print "Column list printed below"
print col_list
print pd.DataFrame(pc_final.components_.T, columns=['PC_' +str(i) for i in range(5)],index=col_list)

print"Factor Analysis : variance explained by each component has thus been focused on "
print pd.Series(pc_final.explained_variance_ratio_,index=['PC_'+ str(i) for i in range(5)])

print type(cr_pca)


##### CLUSTERING #####
print "Beginning with the clustering model . Taking 4 clusters initially"
k4=KMeans(n_clusters=4,random_state=123)
k4.fit(reduced_cr)
print k4.labels_

print pd.Series(k4.labels_).value_counts()

color_map={0:'r',1:'b',2:'g',3:'y'}
label_color=[color_map[l] for l in k4.labels_]
plt.figure(figsize=(7,7))
plt.scatter(reduced_cr[:,0],reduced_cr[:,1],c=label_color,cmap='Spectral',alpha=0.5)
plt.savefig("4 Cluster Plot")

print cr_dummy.dtypes
df_pair_plot=pd.DataFrame(reduced_cr,columns=['PC_' +str(i) for i in range(5)])
df_pair_plot['Cluster']=k4.labels_
plot=sns.pairplot(df_pair_plot,hue='Cluster', palette= 'Dark2', diag_kind='kde',size=1.85)
plot.savefig("6_pairwise relationship of components on the data")

# Key performace variable selection . dropping varibales which are used in derving new KPI
col_kpi=['PURCHASES_TRX','Monthly_avg_purchase','Monthly_cash_advance','limit_usage','CASH_ADVANCE_TRX',
         'payment_minpay','both_oneoff_installment','installment','one_off','none','CREDIT_LIMIT']
print cr_pre.describe()
cluster_df_4=pd.concat([cre_original[col_kpi],pd.Series(k4.labels_,name='Cluster_4')],axis=1)
print "Head of new DF for 4 cluster"
print cluster_df_4.head()

# Mean value gives a good indication of the distribution of data. So we are finding mean value for each variable for each cluster
cluster_4=cluster_df_4.groupby('Cluster_4')\
.apply(lambda x: x[col_kpi].mean()).T
print "FOR 4 CLUSTERS FOLLOWING IS THE DATA"
print cluster_4

fig,ax=plt.subplots(figsize=(15,10))
index=np.arange(len(cluster_4.columns))

cash_advance=np.log(cluster_4.loc['Monthly_cash_advance',:].values)
credit_score=(cluster_4.loc['limit_usage',:].values)
purchase= np.log(cluster_4.loc['Monthly_avg_purchase',:].values)
payment=cluster_4.loc['payment_minpay',:].values
installment=cluster_4.loc['installment',:].values
one_off=cluster_4.loc['one_off',:].values


bar_width=.10
b1=plt.bar(index,cash_advance,color='b',label='Monthly cash advance',width=bar_width)
b2=plt.bar(index+bar_width,credit_score,color='m',label='Credit_score',width=bar_width)
b3=plt.bar(index+2*bar_width,purchase,color='k',label='Avg purchase',width=bar_width)
b4=plt.bar(index+3*bar_width,payment,color='c',label='Payment-minpayment ratio',width=bar_width)
b5=plt.bar(index+4*bar_width,installment,color='r',label='installment',width=bar_width)
b6=plt.bar(index+5*bar_width,one_off,color='g',label='One_off purchase',width=bar_width)

plt.xlabel("Cluster")
plt.title("Insights")
plt.xticks(index + bar_width, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3'))
plt.legend()
plt.savefig("7_4ClusterInsights")

# Percentage of each cluster in the total customer base
print "tabulation of each cluster in relation to customer base"
s=cluster_df_4.groupby('Cluster_4').apply(lambda x: x['Cluster_4'].value_counts())
print s,'\n'

per=pd.Series((s.values.astype('float')/ cluster_df_4.shape[0])*100,name='Percentage')
print "Cluster -4 ",'\n'
print pd.concat([pd.Series(s.values,name='Size'),per],axis=1),'\n'
print "END OF 4 CLUSTERS"

### 5 Clusters
print "WORKING ON 5 CLUSTERS"
k5=KMeans(n_clusters=5,random_state=123)
k5=k5.fit(reduced_cr)
k5.labels_

pd.Series(k5.labels_).value_counts()

plt.figure(figsize=(7,7))
plt.scatter(reduced_cr[:,0],reduced_cr[:,1],c=k5.labels_,cmap='Spectral',alpha=0.5)
plt.xlabel('PC_0')
plt.ylabel('PC_1')
plt.savefig("8_5Clustersmapped")

cluster_df_5=pd.concat([cre_original[col_kpi],pd.Series(k5.labels_,name='Cluster_5')],axis=1)
# Finding Mean of features for each cluster
print "As mean can be used a good representation of each cluster following is the table for the same"
cluster_df_5.groupby('Cluster_5')\
.apply(lambda x: x[col_kpi].mean()).T

s1=cluster_df_5.groupby('Cluster_5').apply(lambda x: x['Cluster_5'].value_counts())
print '\n',s1

# percentage of each cluster
print "Percentage Distribution of each cluster"
print "Cluster-5",'\n'
per_5=pd.Series((s1.values.astype('float')/ cluster_df_5.shape[0])*100,name='Percentage')
print pd.concat([pd.Series(s1.values,name='Size'),per_5],axis=1)

### 6 Clusters

k6=KMeans(n_clusters=6).fit(reduced_cr)
k6.labels_
color_map={0:'r',1:'b',2:'g',3:'c',4:'m',5:'k'}
label_color=[color_map[l] for l in k6.labels_]
plt.figure(figsize=(7,7))
plt.scatter(reduced_cr[:,0],reduced_cr[:,1],c=label_color,cmap='Spectral',alpha=0.5)
plt.savefig("9_6Clustersmapped")

cluster_df_6=pd.concat([cre_original[col_kpi],pd.Series(k6.labels_,name='Cluster_6')],axis=1)

six_cluster=cluster_df_6.groupby('Cluster_6').apply(lambda x: x[col_kpi].mean()).T
print "For the model with Six Clusters below is the data"
print six_cluster

fig,ax=plt.subplots(figsize=(15,10))
index=np.arange(len(six_cluster.columns))

cash_advance=np.log(six_cluster.loc['Monthly_cash_advance',:].values)
credit_score=(six_cluster.loc['limit_usage',:].values)
purchase= np.log(six_cluster.loc['Monthly_avg_purchase',:].values)
payment=six_cluster.loc['payment_minpay',:].values
installment=six_cluster.loc['installment',:].values
one_off=six_cluster.loc['one_off',:].values

bar_width=.10
b1=plt.bar(index,cash_advance,color='b',label='Monthly cash advance',width=bar_width)
b2=plt.bar(index+bar_width,credit_score,color='m',label='Credit_score',width=bar_width)
b3=plt.bar(index+2*bar_width,purchase,color='k',label='Avg purchase',width=bar_width)
b4=plt.bar(index+3*bar_width,payment,color='c',label='Payment-minpayment ratio',width=bar_width)
b5=plt.bar(index+4*bar_width,installment,color='r',label='installment',width=bar_width)
b6=plt.bar(index+5*bar_width,one_off,color='g',label='One_off purchase',width=bar_width)

plt.xlabel("Cluster")
plt.title("Insights")
plt.xticks(index + bar_width, ('Cl-0', 'Cl-1', 'Cl-2', 'Cl-3','Cl-4','Cl-5'))

plt.legend()
plt.savefig("10_6ClusterInsight")

cash_advance=np.log(six_cluster.loc['Monthly_cash_advance',:].values)
credit_score=list(six_cluster.loc['limit_usage',:].values)
print cash_advance

## Checking performance metrics for Kmeans
print "Checking the performance score for the model to identify and validate number of clusters"
score={}
score_c={}
for n in range(3,10):
    kscore=KMeans(n_clusters=n)
    kscore.fit(reduced_cr)
    score[n]=silhouette_score(reduced_cr,kscore.labels_)
print "Silhouette score done"
pd.Series(score).plot()
plt.savefig("11_SilhouetteScoreGraph")
print "MODEL IS COMPLETED"
