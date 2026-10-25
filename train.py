import sys,joblib,json
import pandas as pd,mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report,confusion_matrix
def train(file):
 df=pd.read_csv(file).drop_duplicates("text");train,test=train_test_split(df,test_size=.2,stratify=df.label,random_state=42)
 model=make_pipeline(TfidfVectorizer(ngram_range=(1,2)),CalibratedClassifierCV(LinearSVC(),cv=3))
 with mlflow.start_run(run_name="tfidf-svm"):
  model.fit(train.text,train.label);pred=model.predict(test.text);report=classification_report(test.label,pred,output_dict=True)
  mlflow.log_metric("macro_f1",report["macro avg"]["f1-score"])
  matrix=confusion_matrix(test.label,pred,labels=model.classes_)
  result={"report":report,"labels":model.classes_.tolist(),"confusion_matrix":matrix.tolist()}
  open("evaluation.json","w").write(json.dumps(result,indent=2));joblib.dump(model,"classifier.joblib")
  mlflow.log_artifact("evaluation.json");mlflow.log_artifact("classifier.joblib")
if __name__=="__main__":train(sys.argv[1])
