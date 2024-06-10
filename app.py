from flask import Flask,request,render_template
import pickle
import nltk
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
import pandas as pd

app= Flask(__name__)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
post_stem=PorterStemmer()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def news_prediction(news):
    stemmed_content=re.sub('[^a-zA-Z]',' ',news)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[post_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content

def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(news_prediction) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)
    pred_LR = model.predict(new_xv_test)
    

    return pred_LR[0]

@app.route('/')
def dome():
    return render_template('index.html')

@app.route('/about')
def dome_2():
    return render_template('about.html')

@app.route('/predict' , methods=['GET','POST'])
def prediction():
    if request.method =='POST':  
        news=str(request.form['predict'])
        answer=manual_testing(news)
        print(answer)
        return render_template('index.html',prediction_text=answer,news=news)
       
    else:    
      return render_template("index.html")



 
if __name__=="__main__":
    app.run(debug=True)

