
#importing the required libraries
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
import re
import pdb
import pickle
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#import the dataset
j_df = pd.read_csv('JEOPARDY_CSV.csv')

#dividing the data into response and target variables
X = j_df[' Question']
y = j_df[' Value']

#replace $ with no space to the target variable
y = y.str.replace('$','')
#replace , with no space to the target variable
y = y.str.replace(',','')

#replace none value with zero to the target variable
y[y=='None']= 0

#convert the tarrget column from str to int
y = y.astype(int)


documents = []
stemmer = WordNetLemmatizer()

for sen in range(0,len(X)):
    #remove all the special characters
    document = re.sub(r'\W',' ',str(X[sen]))
    #remove all single characters from the start
    document = re.sub(r'\s+[a-zA-Z]\s+',' ',document)
    #remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ',document)
    #substituting multiple spaces with single space
    document = re.sub(r'\s+',' ',document,flags=re.I)
    #remove links 
    document= re.sub(r'^https?:\/\/.*[\r\n]*', '', document, flags=re.MULTILINE)
    #converting to lower case
    document = document.lower()
    #lemmatization
    document = document.split()
    
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    documents.append(document)
    
#tfidf assumes word might have high frequency of occurrence in other documents as well.
tfidfconverter = TfidfVectorizer(max_features=1500,min_df=5,max_df=0.7,stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents).toarray()

#split the data in to train & test datasets

xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.25,random_state=0)
print(xtrain.shape,ytrain.shape)
print(xtest.shape,ytest.shape)

from sklearn.linear_model import LinearRegression
model_linear = LinearRegression()
model_linear.fit(xtrain,ytrain)

y_pred = model_linear.predict(xtest)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

test_mse = mean_squared_error(ytest,y_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(ytest,y_pred)

from sklearn.model_selection import cross_val_score
cv_result = cross_val_score(model_linear, xtrain, ytrain, cv=4, scoring='neg_mean_squared_error')

