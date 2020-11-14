

# HARSHIL PARIKH

import streamlit as st
import numpy as np
import pandas as pd

# NLP Pkgs
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import os




# Main Stuff

st.title("Greetings NLP - Presence")
st.subheader("Created using Streamlit - Harshil Parikh ")



# Loading the data into streamlit
@st.cache
def load_data(nrows):
    #data = pd.read_csv('/Users/harshilparikh/Desktop/INT/data/selections.csv', nrows=nrows)
    dataset = st.cache(pd.read_csv)('/Users/harshilparikh/Desktop/INT/data/selections.csv')
    return dataset


data_load_state = st.text('Loading data...')
dataset = load_data(1000)
data_load_state.text('Data loaded.')

#Displaying all data first
if st.checkbox('Show Raw data'):
	st.subheader('Raw Data')
st.write(dataset)


# GREETING TAB

st.subheader('Greetings')
greet = st.sidebar.multiselect("Select Greeting", dataset['Greeting'].unique())

select = dataset[(dataset['Greeting'].isin(greet))]
# SEPARATING ONLY TWO COLUMNS FROM THE DATA 
greet_select = select[['Greeting','Selected']]
select_check= st.checkbox("Display records with greeting")
if select_check:
	st.write(greet_select)



#Text- Preprocessing  - Range from 0 to 6758 total feedback
st.subheader('Naive Bayes')
nltk.download('stopwords')
corpus = []
for i in range(0, 6759):
    review = re.sub('[^a-zA-Z]', '', str(dataset['Selected'][i]))
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ''.join(review)
    corpus.append(review)


#BAG OF WORDS
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#Training sets (800 values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#X_train[0, 0:10] #First 10 rows of the first column of X_train.

# NLP - Naive Bayes algorithm 
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred_NB = classifier.predict(X_test)


#Confusion Matrix
cm_NB = confusion_matrix(y_test, y_pred_NB) 
st.write(cm_NB)

#TRUE/FALSE AS PER cm_NB
TN_NB = 5
FN_NB = 668
FP_NB = 7
TP_NB = 672

#CALCULATING ACCURACY OF NB 

Accuracy_NB = (TP_NB + TN_NB) / (TP_NB + TN_NB + FP_NB + FN_NB)
st.subheader('Accuracy of Naive Bayes')
Accuracy_NB

#PRECISION OF NB 
Precision_NB = TP_NB / (TP_NB + FP_NB)
st.subheader('Precision of Naive Bayes')
Precision_NB

#RECALL OF NB
Recall_NB = TP_NB / (TP_NB + FN_NB)
st.subheader('Recall of Naive Bayes')
Recall_NB

#Score of NB
F1_Score_NB = 2 * Precision_NB * Recall_NB / (Precision_NB + Recall_NB) 
st.subheader('Score of Naive Bayes')
F1_Score_NB

# Havent done RandomForest and DecisionTree methods 

