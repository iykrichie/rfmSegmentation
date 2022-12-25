# Databricks notebook source
# MAGIC %md
# MAGIC # RFM Segmentation for retail customers

# COMMAND ----------

# MAGIC %md
# MAGIC Segmentation of non-corporate customers according to their transactional activity. customer data is limited to 6 months transactional history.
# MAGIC 
# MAGIC Variables used:
# MAGIC 
# MAGIC - Transaction Recency - this refers to the total days since the customer carried out a transaction
# MAGIC 
# MAGIC - Transaction Frequency - this refers to the total number of unique transaction done by the customer within the 6-months period
# MAGIC 
# MAGIC - Transaction Monetary Value - this refers to the total amount (value) of transaction done by the customer within the 6- months period.

# COMMAND ----------

# MAGIC %md
# MAGIC ### **Getting required packages**
# MAGIC 
# MAGIC Here we load the packages and required libraries for the project.
# MAGIC 
# MAGIC Note:  I had to comment out lines to to load the _**pyodbc**_ and \_**pandasq**\_l packages since this is just a demo using csv data as input.
# MAGIC 
# MAGIC _<mark>I had loaded the data from MSSQL database and automated a pipleline to repeat this data load every 6-months. Using the job feature on Azure Databricks</mark>_

# COMMAND ----------

#Import Packages
import pandas as pd # working with data
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
import seaborn as sns

#!pip install -U pyodbc
import pyodbc
import sys

## Importing pandasql 
#!pip install -U pandasql  - this was used in connecting to MSSQL database in production, but would not be required for this public use.
import pandasql as psql
from pandasql import sqldf 


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) 

print('Python: ' + sys.version.split('|')[0])
print('Pandas: ' + pd.__version__)
print('pyODBC: ' + pyodbc.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load 6-months transacntion history as input dataset

# COMMAND ----------

query = """


DECLARE @6mnthsAgo datetime2 
DECLARE @Today datetime2 
SET @6mnthsAgo = DATEADD(MONTH, -6, GETUTCDATE()) 
SET @Today = GETUTCDATE()    

select Customer_ID CustId, count(RECID) [Frequency], 
DATEDIFF(DAY, max(BOOKING_DATE), GETDATE()) [Recency],
sum(AMOUNT_LCY - AMOUNT_LCY_SIGN) Monetary, @Today [DateLoaded]

FROM [dbo].[FactStmtTransactions]
LEFT JOIN dbo.dimTransactionCodesT24 d ON transaction_code = d.transactioncode
left outer join dimcustomersfull a on customer_id = a.customerid
left outer join  dbo.dimaccountsv2 c on c.CUSTOMER = a.customerid
left outer join  [edo].[FactAccountProfitability] b on c.accountid = b.accountnumber   

WHERE BOOKING_DATE >= @6mnthsAgo  and a.customer_status in ('1','10','11','12','13','14','16')
and customer_id is not null
AND initiation = 'CUSTOMER'
GROUP BY CUSTOMER_ID;


"""

def dfsqlquery(query):
  # parameters for database connection
  server = 'businessinsight.database.windows.net'
  database = 'SBNEnterpriseDW'
  username = 'edouser'
  password = 'data18@@$$'
  params0 = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password

  # create the connection
  conn1 = pyodbc.connect(params0)
  sql = query
  seg_df = pd.read_sql(sql, conn1)
  return seg_df



df = dfsqlquery(query)

# COMMAND ----------

#import data from csv to dataframe using pandas..
#df = pd.read_csv('dbfs:/user/hive/warehouse/seg_rfm_df')
df.drop_duplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Brief EDA on dataset

# COMMAND ----------

df['Recency'].plot.box() 
plt.show()

# COMMAND ----------

sns.distplot(df['Recency'])
#plt.savefig('plt/DaysSinceLastTx.png')

plt.show()

# COMMAND ----------

sns.distplot(df['Frequency'])
#plt.savefig('plt/txcount.png')

plt.show()

# COMMAND ----------

sns.distplot(df['Monetary'])
#plt.savefig('plt/revenue.png')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing of dataset for modelling

# COMMAND ----------

sdf =  psql.sqldf("select CustId, Recency, Frequency, Monetary from df")
sdf.head()

# COMMAND ----------

rfm = sdf

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clustering by RFM

# COMMAND ----------

quintiles = rfm[['Recency', 'Frequency', 'Monetary']].quantile([.2, .4, .6, .8]).to_dict()
quintiles

# COMMAND ----------

def r_score(x):
    if x <= quintiles['Recency'][.2]:
        return 5
    elif x <= quintiles['Recency'][.4]:
        return 4
    elif x <= quintiles['Recency'][.6]:
        return 3
    elif x <= quintiles['Recency'][.8]:
        return 2
    else:
        return 1

def fm_score(x, c):
    if x <= quintiles[c][.2]:
        return 1
    elif x <= quintiles[c][.4]:
        return 2
    elif x <= quintiles[c][.6]:
        return 3
    elif x <= quintiles[c][.8]:
        return 4
    else:
        return 5 

# COMMAND ----------

rfm['R'] = rfm['Recency'].apply(lambda x: r_score(x))
rfm['F'] = rfm['Frequency'].apply(lambda x: fm_score(x, 'Frequency'))
rfm['M'] = rfm['Monetary'].apply(lambda x: fm_score(x, 'Monetary'))

# COMMAND ----------

rfm['RFM Score'] = rfm['R'].map(str) + rfm['F'].map(str) + rfm['M'].map(str)
rfm.head()

# COMMAND ----------

segt_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Can\'t Loose',
    r'3[1-2]': 'About To Sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'
}

rfm['Segment'] = rfm['R'].map(str) + rfm['F'].map(str)
rfm['Segment'] = rfm['Segment'].replace(segt_map, regex=True)
rfm.head()

# COMMAND ----------

# count the number of customers in each segment
segments_counts = rfm['Segment'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(segments_counts)),
              segments_counts,
              color='silver')
ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
ax.set_yticks(range(len(segments_counts)))
ax.set_yticklabels(segments_counts.index)

for i, bar in enumerate(bars):
        value = bar.get_width()
        if segments_counts.index[i] in ['Champions', 'Loyal Customers']:
            bar.set_color('green')
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                '{:,} ({:}%)'.format(int(value),
                                   int(value*100/segments_counts.sum())),
                va='center',
                ha='left'
               )
#plt.savefig('plt/rfmsegments.png')

plt.show()

# COMMAND ----------

# Distribution of the RFM Segments

sns.distplot(rfm['RFM Score'])
#plt.savefig('plt/rfm_score.png')

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Description of RFM clusters
# MAGIC 
# MAGIC 
# MAGIC | Segment |  Description|
# MAGIC |--|--|
# MAGIC | Champions |  Transacted recently, and often and profitable customers		
# MAGIC |  Loyal Customers|            Our hot high profitable customers, transacting frequently and recently.  |
# MAGIC |Potential Loyalist|Recent customers with average frequency in transaction|
# MAGIC |New Customers	| Recently transacted, but not frequently. |
# MAGIC | Promising |  Recently transacted, but returns low monetary value |
# MAGIC |Customers Needing Attention|Above average recency, frequency and monetary values. May not have transacted very recently                                 though.|
# MAGIC | About To Sleep | Below average recency and frequency. Will lose them if not reactivated |
# MAGIC | At Risk | Below average recency and frequency. Will lose them if not reactivated. |
# MAGIC |Can't Loose|History of frequent transaction but no recency|
# MAGIC | Hibernating | Last transaction was a long time ago |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Processing clustering results

# COMMAND ----------

rfm.head()


# COMMAND ----------

output_table =  psql.sqldf("""
                
    SELECT *, Segment,

    CASE	
		WHEN [RFM Score] >= 500 THEN 'Champion Caleb' 
		WHEN [RFM Score] >= 400 THEN 'Promising Peter' 
		WHEN [RFM Score] >= 300 THEN 'Lukewarm Jude' 
		WHEN [RFM Score] >= 200 THEN 'Cold Clara' 
		WHEN [RFM Score] <200 THEN 'Dormant Dora' 
    END AS [Cluster]
    
   -- Segment



    FROM rfm

                  """)


sdf2 = output_table
sdf2['RunDate'] = pd.to_datetime('today')





# COMMAND ----------

#sdf2.to_csv('clustering_result\rfm_slim_clusters.csv')
sdf2.head()

# COMMAND ----------


