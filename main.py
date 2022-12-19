import uvicorn
from fastapi import FastAPI, Request, Form
import joblib 
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates



app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



vectorizer = joblib.load('Saved_models/tfidf_vectorizer.pkl')
model = joblib.load('Saved_models/pass_agg_model.pkl')



@app.get("/", response_class= HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/predict", response_class= HTMLResponse)
async def predict (request: Request, rawtext: str = Form(...)):
#async def predict (request: Request):    
    #raw_review = await request.form()
    
    if rawtext.strip():
        # Data cleaning and feature extraction
        clean_review = cleanReview(rawtext)
        clean_review_list = [clean_review] # transform review to list for vectorizer only accept list
        
        # TF-IDF vectorization
        tfidf_vec = vectorizer.transform(clean_review_list)

        # Prediction
        predictions = model.predict(tfidf_vec)
        predicted_condition = predictions[0]

        # Recommanded drugs based on the predicted condition
        df = pd.read_csv('Data/drugsComTrain_raw.tsv', sep = "\t")
        top_Drugs = topDrugs(predicted_condition, df)
        
        #return f" {predicted_condition} and {top_Drugs}" 
        #return templates.TemplateResponse('predict.html',rawtext=raw_review,result=predicted_condition,top_drugs=top_Drugs)
        return templates.TemplateResponse('predict.html',{"request": request, "rawtext": rawtext,"result": predicted_condition,"top_Drugs": top_Drugs })

    else:
        #return " No review found! Please enter a review."
        #rawtext ="There is no text to select"
        return templates.TemplateResponse('predict.html',{"request": request, "rawtext": " No review found. Please enter a review!", "result": "No condition can be predicted!.","top_Drugs": "-" })

    




def cleanReview(raw_review):
    # 1. Clean HTML using beautiful soup
    rm_html = BeautifulSoup(raw_review, 'html.parser').get_text()

    # 2. Remove all other irrelevant characters (numbers...) using "re" and replace with a space
    rm_irr_char =  re.sub('[^a-zA-Z]', ' ', rm_html)
    
    # 3. lower the case of all words
    low_case = rm_irr_char.lower().split()
   
    # 4. remove stop words. Use "stopwords" from nltk.corpus
    stop_words = stopwords.words('english')
    rm_stop_words = [word for word in low_case if not word in stop_words]

    # 5. lemmatization using 'WordNetLemmatizer'
    lemmatizer = WordNetLemmatizer()
    lem_words = [lemmatizer.lemmatize(word) for word in rm_stop_words]
    
    # join words and separate them with space
    return ' '.join(lem_words)



def topDrugs(condition, df):
    # Filter the table by selecting the top ratings and usefulcounts
    df_sort = df[(df.rating >= 9) & (df.usefulCount >=100)].sort_values(by = ['rating', 'usefulCount'], ascending=[False, False])

    # get drug names of the condition and select the top 3
    top_Drugs = df_sort[df_sort.condition == condition]["drugName"].head(3).tolist()
    return top_Drugs