import sqlite3,joblib
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel,Field
app=FastAPI()
@lru_cache
def model():return joblib.load("classifier.joblib")
class Document(BaseModel):text:str=Field(min_length=1,max_length=50000)
def route(text,label,confidence,database="review.db"):
 review=confidence<.7
 if review:
  with sqlite3.connect(database) as db:
   db.execute("CREATE TABLE IF NOT EXISTS review(id INTEGER PRIMARY KEY,text TEXT,label TEXT,confidence REAL)")
   db.execute("INSERT INTO review(text,label,confidence) VALUES(?,?,?)",(text,label,confidence))
 return {"label":label,"confidence":confidence,"manual_review":review}
@app.post("/classify")
def classify(d:Document):
 m=model();p=m.predict_proba([d.text])[0];i=int(p.argmax());return route(d.text,str(m.classes_[i]),float(p[i]))
