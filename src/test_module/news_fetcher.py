from newsapi import NewsApiClient
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("src/test_module/data/data.tsv", sep='\t')

train, test = train_test_split(df, test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer()
vectorizer.fit(df['title'])

model = LinearRegression()
model.fit(vectorizer.transform(train['title']), train['score'])

TOKEN = ""    

api = NewsApiClient(TOKEN)
response = api.get_everything(q="apple", language="ru")
for resp in response["articles"]:
    print(resp["title"], model.predict(vectorizer.transform([resp["title"]])))
