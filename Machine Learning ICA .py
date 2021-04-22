#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries needed for project

# In[1]:


#importing libraires I will most likely need
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ### Load in our stroke data as a dataframe

# In[2]:


stroke_data = pd.read_csv('healthcare-dataset-stroke-data.csv')


# In[3]:


pd.set_option('display.max_rows', 200)


# ### Exploratory Data Analysis

# In[4]:


#looking at the first 30 lines of our stroke data
print(stroke_data.head(30))


# In[5]:


###check columns with null values
print(stroke_data.columns[stroke_data.isnull().any()])


# In[6]:


#luckily, only the bmi column has null values, so let's see how many null values there are to give us an idea the best way
#to deal with this problem
print(stroke_data['bmi'].isnull().sum(axis = 0))


# In[7]:


#I don't need the id column so I drop it
new_stroke_df = stroke_data.drop('id', axis=1)


# In[8]:


#number of people with vs without stroke from dataset
print(new_stroke_df['stroke'].value_counts())


# In[9]:


#number of people with vs without heart_disease from dataset
print(new_stroke_df['heart_disease'].value_counts())


# In[10]:


print(new_stroke_df['hypertension'].value_counts())


# In[11]:


print(new_stroke_df['age'].max())


# In[12]:


print(new_stroke_df['age'].min())


# In[13]:


## plot number of people with and without stroke
sns.set_theme(style="darkgrid")
sns.countplot(new_stroke_df['stroke'])


# In[14]:


#view the correlation between numerical attributes in the dataset
print(new_stroke_df.corr())


# In[15]:


#since we have very small percentage of people with stroke from the dataset, it won't be wise to remove rows with null
#bmi values if the stroke value is positive (1). So checking to see what the stroke value was for the NaN bmi's will
#give a better idea what next to do
new_stroke_df[new_stroke_df['bmi'].isnull()& (new_stroke_df['stroke']==1)].describe()


# In[16]:


#I will drop the bmi column since all NaN instances have a stroke value of 1 & it has a very low/ the least correlation 
#with our target varialbe (stroke) as seen in the correlation matrix above
new_stroke_df['bmi'].fillna(value=new_stroke_df['bmi'].mean(),inplace=True)


# In[17]:


#look at what our data looks like now
print(new_stroke_df)


# In[18]:


#double check to make sure no null values
print(new_stroke_df.columns[new_stroke_df.isnull().any()])


# In[19]:


print(new_stroke_df['smoking_status'].value_counts())


# In[20]:


#Smoking status of "Unknown" doesn't help us much but it will also be unwise to randomly replace with one of the other
#smoking status labels since these are the 2nd largest in the dataset with 1544 records, so I check to see how many of
#'Unknown' labels are also stroke positive since the dataset is imbalanced (very few stroke positive samples which is our
#target)
new_stroke_df.loc[(new_stroke_df['smoking_status']=='Unknown')&(new_stroke_df['stroke']==1)].describe()


# In[21]:


sns.set_theme(style="darkgrid")
sns.countplot(new_stroke_df['smoking_status'])


# In[22]:


#see how stroke relates with the other attributes given in the data
for i, column in enumerate(new_stroke_df.columns):
    sns.barplot(x='stroke', y=column, data=new_stroke_df)
    plt.grid()
    plt.show()


# In[23]:


#from the above, we see that stroke is slightly higher amongst males, the higher the age the more prone to stroke people 
#get,those who have suffered hypertension are more likely to stroke, those that have suffered heart disease are more 
#likely to suffer stroke, married people are more likely to suffer stroke, self-employed people are most likely to 
#suffer stroke,those with higer glucose levels are more likely to suffer stroke, those who have formely smoked are more 
#likely to suffer stroke followed by those that smoke and so on as seen from the figures above.


# In[24]:


#using label encoding for the categorical columns with text values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[25]:


#cleaned_stroke_data['age'] = le.fit_transform(cleaned_stroke_data['stroke'])
new_stroke_df['avg_glucose_level'] = le.fit_transform(new_stroke_df['avg_glucose_level'])
new_stroke_df['hypertension'] = le.fit_transform(new_stroke_df['hypertension'])
new_stroke_df['heart_disease'] =le.fit_transform(new_stroke_df['heart_disease'])
new_stroke_df['ever_married'] = le.fit_transform(new_stroke_df['ever_married'])
new_stroke_df['Residence_type'] = le.fit_transform(new_stroke_df['Residence_type'])
new_stroke_df['smoking_status'] =le.fit_transform(new_stroke_df['smoking_status'])
new_stroke_df['work_type'] =le.fit_transform(new_stroke_df['work_type'])
new_stroke_df['gender'] =le.fit_transform(new_stroke_df['gender'])


# In[26]:


print(new_stroke_df)


# In[27]:


#transform the data to be on same scale using sklearn's StandardScaler()


# In[28]:


from sklearn.preprocessing import StandardScaler


# In[29]:


scale = StandardScaler()


# In[30]:


X = new_stroke_df.drop('stroke',axis=1)
y = new_stroke_df['stroke']


# In[31]:


print(X)


# In[32]:


X = scale.fit_transform(X)


# In[33]:


print(X)


# In[34]:


#splitting the data into my train and test set in a stratified fashion using the stroke labels so there is a well
#distributed proportion of stroke labels in each set since there are few stroke positive samples compared to the rest of 
#the samples.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


# In[35]:


#so we can see how many of each instances of the stroke's labels are in our dependent variable
print(np.unique(y_train, return_counts=True))


# In[36]:


#We see 47 samples have a smoking status of unknown which doesn't give us the information we want about their smoking
#status. I am going to use an oversampling technique called SMOTE since this dataset is very imbalanced with regard
#and drop this rows with 'unknown' smoking status
from imblearn.over_sampling import SMOTE


# In[37]:


#using SMOTE to resample the training set
smt = SMOTE(random_state=42)
X_train_sm,y_train_sm = smt.fit_resample(X_train,y_train)


# In[38]:


#so we can see how many instance of each class labels in our dependent variable after SMOTE
print(np.unique(y_train_sm, return_counts=True))


# In[39]:


model = LogisticRegression().fit(X_train_sm,y_train_sm)


# In[40]:


#using the trained model on the test set
pred_y = model.predict(X_test)


# In[41]:


print("Score on training set: {}".format(model.score(X_train,y_train)))
print("Score on test set: {}".format(model.score(X_test,y_test)))


# In[42]:


#using classification report to also view more important metrics such as precision and recall,
#we can see the logistic regression did quite well
print("Claasification report")
print(classification_report(y_test, pred_y))


# In[43]:


#The logistic regression model has a an ok score of 0.74 from the training data & 0.758 on the test
#on the test set


# In[44]:


#using a confusion matrix to see the number of correctly classified samples for our test data
from sklearn.metrics import confusion_matrix


# In[45]:


cm = confusion_matrix(y_test, pred_y)
print(np.unique(y_test, return_counts=True))
print(np.unique(pred_y, return_counts=True))
print(cm)


# In[46]:


#we can see that we have 735 True negatives (no stroke) and 40 True postives (stroke) and a low false negative
#which is quite good given we don't want to miss knowing patients likely to have stroke so we could warn them
#Using a heatmap to make the results from the confusion matrix of our logistic model more readable
sns.heatmap(cm,square=True,annot=True,fmt='d',cbar='True')
plt.xlabel('predicted values')
plt.ylabel('true values')


# In[77]:


#Would use cross validation here as random forests have a lot of hyperparameters in order to make sure the model would 
#generalize well and also help with tuning parameters
from sklearn.model_selection import cross_val_score


# In[85]:


#Using a random forest classifier with 10 trees after parameter tuning with the help of cross_validation and 
#enabling bootstrapping so we use different combinations of features to build each tree
rf_clf = RandomForestClassifier(n_estimators=10, bootstrap=True,max_depth=16, random_state=0)


# In[90]:


#using 5 fold cross-validation
scores = cross_val_score(rf_clf,X_train_sm,y_train_sm,cv=5)


# In[91]:


print(scores)


# In[92]:


#checking the average accuracy from our cross validation and standard deviation
#we have a low standard deviation which is a good indicator this model performs well
print("%0.2f accuracy and standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[93]:


rf_clf.fit(X_train_sm,y_train_sm)


# In[94]:


print("Score on training set: {}".format(rf_clf.score(X_train,y_train)))
print("Score on test set: {}".format(rf_clf.score(X_test,y_test)))


# In[50]:


y_pred = rf_clf.predict(X_test)


# In[51]:


#we can see that our random forest model has a slightly better precision but a far worse recall on 
#the stroke positive class  
print("Claasification report")
print(classification_report(y_test, y_pred))


# In[52]:


#using a confusion matrix to see how many of our predictions were correct on the test set using random forest
rf_cm = confusion_matrix(y_test, y_pred)
print(np.unique(y_test, return_counts=True))
print(np.unique(y_pred, return_counts=True))
print(rf_cm)


# In[53]:


#using a heatmap to make the random forest's confusion matrix for the test set more readable
sns.heatmap(rf_cm,square=True,annot=True,fmt='d',cbar='True')
plt.xlabel('predicted values')
plt.ylabel('true values');


# In[54]:


#from the above, we see that the random forest generalizes well and correctly identifies 5 stroke patients on the data it
#hasn't been exposed to before. The model also correctly identifies 970 non-stroke patients.This is considered good 
#because from the original dataset 95.1% of the samples don't have stroke.

#We also see it does very well on the 


# In[55]:


#Using KNN


# In[56]:


from sklearn.neighbors import KNeighborsClassifier


# In[57]:


#although there is no clear cut out way to find the optimal value of k, I will try k within the range 3 to 13 and pick the
#value of k with the best result for this problem
neighbors = np.arange(3,13)

#fig, axes = plt.subplots(2,5,figsize=(10,3))

for k in neighbors:
    knn_model = KNeighborsClassifier(n_neighbors = k)
    knn_model.fit(X_train_sm,y_train_sm)
    knn_y_pred = knn_model.predict(X_test)
    y_ =  knn_model.predict(X_train)
    knn_cm = confusion_matrix(y_test, knn_y_pred)

    print("when k is {}".format(k))
    print("test set confusion matrix")
    print(knn_cm)
    print("training set accuracy is {}:".format(knn_model.score(X_train,y_train)))
    print("test set accuracy is: {}".format(knn_model.score(X_test,y_test)))
    print("Claasification report")
    print(classification_report(y_test, knn_y_pred))
    print()


# In[58]:


#we see from the output above, our results vary quite a bit based on the value of k. In choosing the value for k, since we
#have 2 classes, I decided to go for an odd value of k that generalizes well both on the training and test set although you
#can also break ties when they happen. I would go for k=5 as it generalizes quite well on the data and has a lower false 
#negative than k=3 since what we our looking at predicting is catching the possibilities of stroke and we don't want 
#to miss this in patients it could happen to so there is the possiblity of preventing its occurence
chosen_knn_model = KNeighborsClassifier(n_neighbors = 5)
chosen_knn_model.fit(X_train_sm,y_train_sm)
knn_y_pred = chosen_knn_model.predict(X_test)
y_ =  knn_model.predict(X_train)
knn_cm = confusion_matrix(y_test, knn_y_pred)
knn_cm


# In[59]:


#using a heatmap to make the knn's confusion matrix for the test set more readable
sns.heatmap(knn_cm,square=True,annot=True,fmt='d',cbar='True')
plt.xlabel('predicted values')
plt.ylabel('true values')

