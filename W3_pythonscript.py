# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # PREDICTING CAR ACCIDENT SEVERITY
# 
# ## IBM Applied Data Science Capstone Project
# 
# 
# %% [markdown]
# ## 1. Introduction / Business Problem
# %% [markdown]
# In a big city where car accidents happen all the time, it can be a challenge to deploy necessary number or type of personnel on time with the limited numbers of personnel on our disposal.
# 
# The idea is to classify the severity of a car accident, in this case we will use two level of severity, 1 for Property Damage Only Collision and 2 for Injury Collision. The severity prediction will be based on the information received at the time an accident is reported.
# 
# With this simplification of early accident classification, the Dispatch Center can decide which personnel should be dispatched for the accident. For example, for accident with severity of 1 Property Damage Only Collision, the healthcare personnels are not needed on site, and they can be allocated to another injury related accident.
# %% [markdown]
# ## 2. Data
# %% [markdown]
# The data that will be used is to approach the problem is the sample data set from:
# 
# https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Data-Collisions.csv. 
# 
# This is a Seattle's car accident data from 2004 to 2020 which contains a number of information for each accident, such as the time, location, and the number of people / vehicle involved in each accident. Based on this historical data, we will try to build a model that is able to predict the severity of an accident based on the initial data collected from the accident site.
# 
# The data itself containing 1 target column & 37 feature columns, some of them are not neccessarily useful for us in building the model.
# 
# The target column is **SEVERITYCODE** which contains the severity classification. We have 2 different severity values here:
# 
# - 1 Property Damage Only Collision
# - 2 Injury Collision
# 
# These are the feature columns.
# 
#     'X', 'Y', 'OBJECTID', 'INCKEY', 'COLDETKEY', 'REPORTNO',
#     'STATUS', 'ADDRTYPE', 'INTKEY', 'LOCATION', 'EXCEPTRSNCODE',
#     'EXCEPTRSNDESC', 'SEVERITYCODE.1', 'SEVERITYDESC', 'COLLISIONTYPE',
#     'PERSONCOUNT', 'PEDCOUNT', 'PEDCYLCOUNT', 'VEHCOUNT', 'INCDATE',
#     'INCDTTM', 'JUNCTIONTYPE', 'SDOT_COLCODE', 'SDOT_COLDESC',
#     'INATTENTIONIND', 'UNDERINFL', 'WEATHER', 'ROADCOND', 'LIGHTCOND',
#     'PEDROWNOTGRNT', 'SDOTCOLNUM', 'SPEEDING', 'ST_COLCODE', 'ST_COLDESC',
#     'SEGLANEKEY', 'CROSSWALKKEY', 'HITPARKEDCAR'
# 
# The explanation for each column can be found in:
# 
# https://s3.us.cloud-object-storage.appdomain.cloud/cf-courses-data/CognitiveClass/DP0701EN/version-2/Metadata.pdf
# 
# We exclude the columns that are entered by the state as they won't be available in the initial report ('PEDCOUNT', 'PEDCYLCOUNT', 'VEHCOUNT', 'INJURIES', 'SERIOUSINJURIES', 'FATALITIES'). We also exclude the 'LOCATION' column as this is a free text column and is already represented by the coordinates('X', 'Y').
# 
# We are going to use the following feature columns in our initial model and adding or remove the features as necessary as we build the model.
# 
# - X                 - Double    - Longitude
# - Y                 - Double    - Latitude                
# - ADDRTYPE          - Text, 12  - Collision address type: Alley, Block, Intersection
# - INTKEY            - Double    - Key that corresponds to the intersection associated with a collision 
# - PERSONCOUNT       - Double    - The total number of people involved in the collision
# - SDOT_COLCODE      - Text, 10  - A code given to the collision by SDOT.
# - INATTENTIONIND    - Text, 1   - Whether or not collision was due to inattention. (Y/N) 
# - UNDERINFL         - Text, 10  - Whether or not a driver involved was under the influence of drugs or alcohol. 
# - WEATHER           - Text, 300 - A description of the weather conditions during the time of the collision. 
# - ROADCOND          - Text, 300 - The condition of the road during the collision. 
# - LIGHTCOND         - Text, 300 - The light conditions during the collision. 
# - SPEEDING          - Text, 1   - Whether or not speeding was a factor in the collision. (Y/N)
# - ST_COLCODE        - Text, 10  - A code provided by the state that describes the collision. See the State Collision Code Dictionary in the Metadata file. 
# - SEGLANEKEY        - Long      - A key for the lane segment in which the collision occurred. 
# - CROSSWALKKEY      - Long      - A key for the crosswalk at which the collision occurred. 
# - HITPARKEDCAR      - Text, 1   - Whether or not the collision involved hitting a parked car. (Y/N) 
# %% [markdown]
# ### 2.a. Importing Data

# %%
path = "./DATA/Data-Collisions.csv"


# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
data = pd.read_csv(path)


# %%
data.head()


# %%
data.columns


# %%
data.info()

# %% [markdown]
# We have 194673 car accident records, some of them seems to be missing some information.

# %%
len(data)


# %%
sns.countplot(pre_data['SEVERITYCODE'])

# %% [markdown]
# Looks like the dataset is very unbalanced and skewed to SEVERITYCODE 1. We'll need to address this before training our model later.
# %% [markdown]
# ### 2.b. Create a copy of the data for preprocessing

# %%
pre_data = data[['SEVERITYCODE', 'X', 'Y', 'ADDRTYPE', 'INTKEY', 'PERSONCOUNT','SDOT_COLCODE','INATTENTIONIND', 'UNDERINFL', 'WEATHER', 'ROADCOND', 'LIGHTCOND', 'SPEEDING', 'ST_COLCODE', 'SEGLANEKEY', 'CROSSWALKKEY', 'HITPARKEDCAR']].copy()
pre_data.head()

# %% [markdown]
# ### 2.c. Check for Missing Values

# %%
pre_data.isna().sum()

# %% [markdown]
# ### 2.d. Cleaning up the Data
# %% [markdown]
# We'll go through each columns one be by one to see if there is any necessary actions needed to clean up the data
# %% [markdown]
# #### X, Y
# %% [markdown]
# There are 5334 lines without coordinates data.

# %%
pre_data[['X', 'Y']].isna().sum()

# %% [markdown]
# Let's try to plot the coordinates and differentiate them by their SEVERITYCODE.

# %%
sns.scatterplot(x = pre_data['X'], y = pre_data['Y'], hue = pre_data['SEVERITYCODE'].tolist(), palette = 'deep')

# %% [markdown]
# There doesn't seem to be a clear separation between SEVERITYCODE 1 & 2 based on the coordinates. We'll leave them as is for now.
# %% [markdown]
# ### ADDRTYPE
# Text, 12 - Collision address type:
# * Alley
# * Block
# * Intersection

# %%
pre_data['ADDRTYPE'].unique()


# %%
pre_data['ADDRTYPE'].value_counts(dropna = False)


# %%
sns.countplot(x = 'ADDRTYPE', data = pre_data, hue = 'SEVERITYCODE')

# %% [markdown]
# From the graph above we can see that we have more SEVERITYCODE 1 when the accident is happened in the blocks. 
# %% [markdown]
# Further investigation shows that we are missing coordinates data for all Alley accidents.

# %%
pre_data[pre_data['ADDRTYPE'] == 'Alley'][['ADDRTYPE', 'X', 'Y']].value_counts()

# %% [markdown]
# Since we only have 1926 missing data from ADDRTYPE and 5334 missing data from X and Y, also since ADDRTYPE seems more related to SEVERITYCODE, we'll drop X and Y columns along with the rows with missing ADDRTYPE from our dataframe.

# %%
pre_data.drop(['X', 'Y'], axis = 1, inplace = True)
pre_data.dropna(subset = ['ADDRTYPE'], inplace = True)

# %% [markdown]
# Let's check our data again.

# %%
pre_data.head()


# %%
pre_data.info()


# %%
pre_data.isna().sum()

# %% [markdown]
# #### INTKEY
# Double - Key that corresponds to the intersection associated with a collision 

# %%
pre_data[['INTKEY']].info()


# %%
pre_data['INTKEY'].isna().sum()


# %%
pre_data['INTKEY'].unique()

# %% [markdown]
# INTKEY refers to intersection number related to the acccident. Since more than half of the information are missing, we'll drop this column

# %%
pre_data.drop('INTKEY', axis = 1, inplace = True)

# %% [markdown]
# #### PERSONCOUNT
# Double - The total number of people involved in the collision

# %%
pre_data['PERSONCOUNT'].unique()


# %%
pre_data['PERSONCOUNT'].isna().sum()


# %%
plt.figure(figsize = (20,4))
sns.countplot(pre_data['PERSONCOUNT'], hue = pre_data['SEVERITYCODE'])

# %% [markdown]
# PERSONCOUNT looks good, nothing to be done. 
# %% [markdown]
# #### SDOT_COLCODE
# Text, 10 - A code given to the collision by SDOT.

# %%
pre_data['SDOT_COLCODE'].unique()


# %%
pre_data['SDOT_COLCODE'].isna().sum()


# %%
plt.figure(figsize = (20,4))
sns.countplot(pre_data['SDOT_COLCODE'], hue = pre_data['SEVERITYCODE'])

# %% [markdown]
# SDOT_COLCODE looks good, nothing to be done. 
# %% [markdown]
# #### INATTENTIONIND
# Text, 1 - Whether or not collision was due to inattention. (Y/N) 
# 

# %%
pre_data['INATTENTIONIND'].unique()


# %%
pre_data['INATTENTIONIND'].isna().sum()

# %% [markdown]
# We'll clean up INATTENTIONIND by replacing NaN with 0 and 'Y' with 1

# %%
pre_data['INATTENTIONIND'].replace([np.nan, 'Y'], [0,1], inplace = True)

# %% [markdown]
# We can see that more SEVERITYCODE 1 mostly happens when INATTENTIONIND = 0. 

# %%
sns.countplot(pre_data['INATTENTIONIND'], hue = pre_data['SEVERITYCODE'])

# %% [markdown]
# #### UNDERINFL
# Text, 10 - Whether or not a driver involved was under the influence of drugs or alcohol. 

# %%
pre_data['UNDERINFL'].unique()


# %%
pre_data['UNDERINFL'].value_counts(dropna = False)

# %% [markdown]
# We'll clean up UNDERINFL by replaceing [[NaN, 'N', '0']] with 0 and [['Y', '1']] with 1.

# %%
pre_data['UNDERINFL'].replace(['N', '0', np.nan, '1', 'Y'], [0, 0, 0, 1, 1], inplace = True)

# %% [markdown]
# We can see that more SEVERITYCODE 1 happens when UNDERINFL is 0.

# %%
sns.countplot(pre_data['UNDERINFL'], hue = pre_data['SEVERITYCODE'])

# %% [markdown]
# #### WEATHER
# Text, 300 - A description of the weather conditions during the time of the collision.

# %%
pre_data['WEATHER'].unique()


# %%
pre_data['WEATHER'].value_counts(dropna = False)

# %% [markdown]
# We'll group together NaN, Unknown, Other as Other.

# %%
pre_data['WEATHER'].replace([np.nan, 'Unknown'], ['Other', 'Other'], inplace = True)

pre_data['WEATHER'].unique()


# %%
pre_data['WEATHER'].value_counts()

# %% [markdown]
# Interestingly, most of the accidents happen during clear weather.

# %%
plt.figure(figsize = (20,4))
sns.countplot(pre_data['WEATHER'], hue = pre_data['SEVERITYCODE'])

# %% [markdown]
# #### ROADCOND
# Text, 300 - The condition of the road during the collision.

# %%
pre_data['ROADCOND'].unique()


# %%
pre_data['ROADCOND'].value_counts(dropna = False)

# %% [markdown]
# There are some values that can be grouped together:
# * Wet (Wet, Standing Water)
# * Dry
# * Other (nan, Unknown, Other)
# * Snow/Ice (Snow/Slush, Ice)
# * Sand/Mud/Dirt
# * Oil

# %%
pre_data['ROADCOND'].replace(['Standing Water', np.nan, 'Unknown', 'Snow/Slush', 'Ice'], ['Wet', 'Other', 'Other', 'Snow/Ice', 'Snow/Ice'], inplace = True)


# %%
pre_data['ROADCOND'].unique()


# %%
pre_data['ROADCOND'].value_counts()

# %% [markdown]
# Looks like most of the car accidents happen when the road is dry.

# %%
plt.figure(figsize = (20,4))
sns.countplot(pre_data['ROADCOND'], hue = pre_data['SEVERITYCODE'])

# %% [markdown]
# #### LIGHTCOND
# Text, 300 - The light conditions during the collision. 

# %%
pre_data['LIGHTCOND'].unique()


# %%
pre_data['LIGHTCOND'].value_counts(dropna = False)

# %% [markdown]
# There are some values that can be grouped together:
# * Daylight
# * Dark (Dark - Street Lights On, Dark - No Street Lights, Dark - Street Lights Off, Dark - Unknown Lighting)
# * Dusk
# * Dawn
# * Other (nan, Other, Unknown)

# %%
pre_data['LIGHTCOND'].replace(['Dark - Street Lights On', 'Dark - No Street Lights', 'Dark - Street Lights Off', 'Dark - Unknown Lighting', np.nan, 'Unknown'], 
['Dark', 'Dark', 'Dark', 'Dark', 'Other', 'Other'], inplace = True)

pre_data['LIGHTCOND'].unique()


# %%
pre_data['LIGHTCOND'].value_counts()

# %% [markdown]
# It's interesting that most of the accidents happened when the light condition is good (Daylight).

# %%
plt.figure(figsize = (20,4))
sns.countplot(pre_data['LIGHTCOND'], hue = pre_data['SEVERITYCODE'])

# %% [markdown]
# #### SPEEDING
# Text, 1 - Whether or not speeding was a factor in the collision. (Y/N)

# %%
pre_data['SPEEDING'].unique()


# %%
pre_data['SPEEDING'].value_counts(dropna = False)

# %% [markdown]
# We'll convert SPEEDING into binary data.

# %%
pre_data['SPEEDING'].replace([np.nan, 'Y'], [0, 1], inplace = True)
pre_data['SPEEDING'].unique()


# %%
pre_data['SPEEDING'].value_counts(dropna = False)


# %%
sns.countplot(pre_data['SPEEDING'], hue = pre_data['SEVERITYCODE'])

# %% [markdown]
# #### ST_COLCODE
# Text, 10 - A code provided by the state that describes the collision.
# 
# For more information about these codes, please see the State Collision Code Dictionary. 

# %%
pre_data['ST_COLCODE'].unique()

# %% [markdown]
# We can see that there are 18 missing data for ST_COLCODE. Since this is an insignificant number compared to the total data, we'll remove the lines with missing ST_COLCODE info.

# %%
pre_data['ST_COLCODE'].isna().sum()


# %%
pre_data.dropna(subset = ['ST_COLCODE'], inplace = True)
pre_data['ST_COLCODE'].isna().sum()

# %% [markdown]
# Next we will remove the lines with ' ' as their value in the ST_COLCODE column. There's a total of 4779 rows of them, which is still not as significant compared to the number of data we have.

# %%
pre_data[pre_data['ST_COLCODE'] == ' ']['ST_COLCODE'].count()


# %%
pre_data.drop(pre_data.index[pre_data['ST_COLCODE'] == ' '], inplace = True)


# %%
len(pre_data)

# %% [markdown]
# Next we will convert ST_COLCODE to int to make it easier when building the model.

# %%
pre_data['ST_COLCODE'] = pre_data['ST_COLCODE'].astype('int64')


# %%
plt.figure(figsize = (30,4))
sns.countplot(pre_data['ST_COLCODE'], hue = pre_data['SEVERITYCODE'])

# %% [markdown]
# #### SEGLANEKEY
# Long - A key for the lane segment in which the collision occurred. 

# %%
pre_data['SEGLANEKEY'].unique()


# %%
pre_data['SEGLANEKEY'].isna().sum()


# %%
pre_data[pre_data['SEGLANEKEY'] == 0]['SEGLANEKEY'].count()

# %% [markdown]
# We have quite a lot of missing SEGLANEKEY information. We'll drop this column.

# %%
pre_data.drop('SEGLANEKEY', axis = 1, inplace = True)

# %% [markdown]
# #### CROSSWALKKEY
# Long - A key for the crosswalk at which the collision occurred. 

# %%
pre_data['CROSSWALKKEY'].unique()


# %%
pre_data['CROSSWALKKEY'].isna().sum()


# %%
pre_data[pre_data['CROSSWALKKEY'] == 0]['CROSSWALKKEY'].count()

# %% [markdown]
# Again there is a lot of rows with missing CROSSWALKKEY (0), so we'll drop this column too.

# %%
pre_data.drop('CROSSWALKKEY', axis = 1, inplace = True)

# %% [markdown]
# #### HITPARKEDCAR
# Text, 1 - Whether or not the collision involved hitting a parked car. (Y/N) 

# %%
pre_data['HITPARKEDCAR'].unique()

# %% [markdown]
# We'll convert HITPARKEDCAR into binary data by replacing the values with 0 and 1.

# %%


pre_data['HITPARKEDCAR'].replace(['N', 'Y'], [0, 1], inplace = True)
pre_data['HITPARKEDCAR'].unique()


# %%
sns.countplot(pre_data['HITPARKEDCAR'], hue = pre_data['SEVERITYCODE'])

# %% [markdown]
# #### Let's review the data again

# %%
pre_data.head()


# %%
len(pre_data)


# %%
pre_data.info()

# %% [markdown]
# 
# %% [markdown]
# ## 3. Preparing the Data for Training
# %% [markdown]
# #### 3.a. One-Hot Encoding
# %% [markdown]
# First, we need to encode categorical features WEATHER, ROADCOND, and LIGHTCOND into numerical values using one-hot encoding technique. We'll use get_dummies function from pandas package for this.

# %%
addrtype_dummy = pd.get_dummies(pre_data['ADDRTYPE']).drop('Alley', axis = 1)
weather_dummy = pd.get_dummies(pre_data['WEATHER']).drop('Other', axis = 1)
roadcond_dummy = pd.get_dummies(pre_data['ROADCOND']).drop('Other', axis = 1)
lightcond_dummy = pd.get_dummies(pre_data['LIGHTCOND']).drop('Other', axis = 1)


# %%
pre_data = pd.concat([pre_data, addrtype_dummy, weather_dummy, roadcond_dummy, lightcond_dummy], axis = 1)
pre_data.head()

# %% [markdown]
# We'll drop ADDRTYPE, WEATHER, ROADCOND, and LIGHTCOND since we already have generated the dummy features from them.

# %%
pre_data.drop(['ADDRTYPE', 'WEATHER', 'ROADCOND', 'LIGHTCOND'], axis = 1, inplace = True)
pre_data.head()

# %% [markdown]
# ### 3.b. Train, Test Split

# %%


# %% [markdown]
# Now we'll split the data into training dataset and test dataset using test_train_split function.

# %%
X = pre_data.loc[:,'PERSONCOUNT':]
y = pre_data['SEVERITYCODE']


# %%
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# %%
print('X_train.shape() = ', X_train.shape, ', y_train.shape() = ', y_train.shape)
print('X_test.shape()  = ', X_test.shape, ', y_test.shape()  = ', y_test.shape)


# %%
sns.countplot(pre_data['SEVERITYCODE'])


# %%
sns.countplot(y_train)


# %%
sns.countplot(y_test)

# %% [markdown]
# Since the data is skewed toward, SEVERITYCODE = 1, we will upsample the data for SEVERITYCODE = 2. We will do this for our training dataset.
# 
# First we will need to recombine X_train and y_train using pd.concat.

# %%

X_train = pd.concat([X_train, y_train], axis = 1)
X_train.head()


# %%
print('SEVERITYCODE 1 = ',X_train[X_train['SEVERITYCODE'] == 1]['SEVERITYCODE'].count())
print('SEVERITYCODE 2 = ',X_train[X_train['SEVERITYCODE'] == 2]['SEVERITYCODE'].count())

# %% [markdown]
# Then we will upsample the data for SEVERITYCODE 2 using resample function from sklearn.

# %%
from sklearn.utils import resample

X_1 = X_train[X_train['SEVERITYCODE'] == 1]
X_2 = X_train[X_train['SEVERITYCODE'] == 2]

X_2_upsample = resample(X_2, replace=True, n_samples=len(X_1), random_state=42)
len(X_2_upsample)

# %% [markdown]
# Next we combine X_1 and X_2_upsample.

# %%
X_train_upsample = pd.concat([X_1, X_2_upsample], axis = 0)
len(X_train_upsample)

# %% [markdown]
# And split it again into X_train and y_train.

# %%
y_train_upsample = X_train_upsample['SEVERITYCODE']
X_train_upsample.drop('SEVERITYCODE', axis = 1, inplace = True)


# %%
y_train_upsample


# %%
X_train_upsample

# %% [markdown]
# ## 4. Model Building
# %% [markdown]
# Due to computanional limitation, we will only use Logistic Regression, Decision Tree, and Support Vector Machine for the models. 
# %% [markdown]
# ### 4.a. Logistic Regression
# 

# %%
from sklearn.linear_model import LogisticRegression

mod_log_r = LogisticRegression()
mod_log_r.fit(X_train_upsample, y_train_upsample)
yhat_log_r = mod_log_r.predict(X_test)
yhat_log_r_proba = mod_log_r.predict_proba(X_test)
print("Logistic Regression's Accuracy: ", metrics.accuracy_score(y_test, yhat_log_r))


# %%


# %% [markdown]
# ### 4.b. Decision Tree

# %%
from sklearn.tree import DecisionTreeClassifier

mod_tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4).fit(X_train_upsample, y_train_upsample)
yhat_tree = mod_tree.predict(X_test)
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_test, yhat_tree))

# %% [markdown]
# ### 4.c. Support Vector Machine

# %%
from sklearn import svm

mod_svm = svm.SVC(kernel='rbf', gamma = 'scale').fit(X_train_upsample, y_train_upsample)
yhat_svm = mod_svm.predict(X_test)
print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_test, yhat_svm))

# %% [markdown]
# ## 5. Model Evaluation

# %%
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score

report = pd.DataFrame(index = ['LogisticRegression', 'Decision Tree', 'SVM'], columns = ['Jaccard', 'F1-score'])

report.loc['LogisticRegression', 'Jaccard'] = jaccard_score(y_test, yhat_log_r)
report.loc['LogisticRegression', 'F1-score'] = f1_score(y_test, yhat_log_r, average = 'weighted')

report.loc['Decision Tree', 'Jaccard'] = jaccard_score(y_test, yhat_tree)
report.loc['Decision Tree', 'F1-score'] = f1_score(y_test, yhat_tree, average = 'weighted')

report.loc['SVM', 'Jaccard'] = jaccard_score(y_test, yhat_svm)
report.loc['SVM', 'F1-score'] = f1_score(y_test, yhat_svm, average = 'weighted')


report.index.name = 'Algorithm'
report

# %% [markdown]
# Based on the scores, Decision Tree has the best performance out of all the three models that we build. 
# %% [markdown]
# We can also print out the confusion matrixes for each model.

# %%
print('Confusion Matrix for Logistic Regression:')
print(metrics.confusion_matrix(y_test, yhat_log_r))
print()
print('Confusion Matrix for Decision Tree:')
print(metrics.confusion_matrix(y_test, yhat_tree))
print()
print('Confusion Matrix for Support Vector Machine:')
print(metrics.confusion_matrix(y_test, yhat_svm))

# %% [markdown]
# ## 6. Conclusion
# %% [markdown]
# The data we use have an unbalanced number of SEVERITYCODE values and is heavily skewed toward SEVERITYCODE 1. Also, some of the lines are missing some information. Due to those, we needed to remove rows with missing information and resampled the training data to reinforce the signal of the data in the minor category (SEVERITYCODE 2).
# 
# With those limitations, we managed to build three classification models, Logistic Regression, Decision Tree, and Support Vector Machine, even though there are still room to improve the model’s performances.
# 
# Comparing the scores for those models, we have the Decision Tree model that gives us the best accuracy score. With better dataset, we sure can improve the performance of the model.

# %%



