import streamlit as st  #used for streamlit api reference 
# below all libraries were part of SMS Spam classifier model building and hence add them again. 
import pickle     #to load the saved pickle files 
import pandas as pd 
import numpy as np 
import string 
import nltk      #natural language tool kit used for text processing 
#nltk.data.path.append('./nltk_data')
from nltk.corpus import stopwords  #text processing 
import string 
from nltk.stem.porter import PorterStemmer  #text processing 
import pandas as pd 
ps=PorterStemmer()    
from xgboost import XGBClassifier 

# nltk.download('punkt', download_dir='nltk_data')
# nltk.download('stopwords', download_dir='nltk_data')

try:
    nltk.data.find('nltk_data/tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Check if 'stopwords' is downloaded
try:
    nltk.data.find('nltk_data/corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

#function to convert SMS text to numerical form ,SMS text  we will receive on our deployed app to predict. 
def transform_text(text): 
        text=text.lower() 
        y=[] 
        #tokenization 
        text=nltk.word_tokenize(text) 
        for i in text: 
            if i.isalnum(): 
                y.append(i) 
        text=y[:] 
        y.clear() 
        #removing stopwords and punctuations 
        for i in text: 
            if i not in stopwords.words('english') and i not in string.punctuation: 
                y.append(i) 
        text=y[:] 
        y.clear() 
     
        #stemming applied on text 
        for i in text: 
            y.append(ps.stem(i)) 
        return y 
#loading  both the models from respective directory 
tfidf=pickle.load(open('vectorizer.pkl','rb')) 
model=pickle.load(open('mnb_spam_detector.pkl','rb')) 
#streamlit app title 
st.title("SMS Spam classifier")         
input_sms= st.text_area("Enter the message") 
 
#predict button , when clicked will execute the process 
if st.button('Predict'): 
 
#1.preprocess- converting the input_sms received by user on app   
 
    transform_sms=transform_text(input_sms) 
    print(type(transform_sms)) 
    transform_sms=np.array(transform_sms) #converting the list of string format to array of string format 
 
#2.vectorize - converting the received text SMS into numeric for model understanding 
 
    vector_input=tfidf.transform(transform_sms.astype('str')).toarray() 
    print(type(vector_input)) 
    print(vector_input) 
    vector_input = pd.DataFrame(vector_input,columns=tfidf.get_feature_names_out()) 
 
#3.predict - passing the converted text to model to predict if it is spam or ham 
 
    prediction= model.predict(vector_input)[0] 
 
#4.display- the result on app itself , if prediction result is 1 then ui(button) will display  Spam else Not Spam 
     
    if prediction==1: 
        st.header("Spam") 
    else: 
        st.header("Not Spam") 
