import sys,numpy as np,pandas as pd,mlflow
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments
df=pd.read_csv(sys.argv[1]).drop_duplicates("text");labels=sorted(df.label.unique());df["label"]=df.label.map({x:i for i,x in enumerate(labels)})
a,b=train_test_split(df,test_size=.2,stratify=df.label,random_state=42)
tokenizer=AutoTokenizer.from_pretrained("distilbert-base-uncased")
def dataset(frame):return Dataset.from_pandas(frame[["text","label"]],preserve_index=False).map(lambda batch:tokenizer(batch["text"],truncation=True,padding="max_length",max_length=256),batched=True)
model=AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=len(labels),id2label=dict(enumerate(labels)),label2id={x:i for i,x in enumerate(labels)})
def metrics(output):
 prediction=np.argmax(output.predictions,axis=-1)
 return {"accuracy":accuracy_score(output.label_ids,prediction),"macro_f1":f1_score(output.label_ids,prediction,average="macro")}
with mlflow.start_run(run_name="distilbert"):
 trainer=Trainer(model=model,args=TrainingArguments(output_dir="transformer-model",num_train_epochs=2,per_device_train_batch_size=8,eval_strategy="epoch",report_to=[]),train_dataset=dataset(a),eval_dataset=dataset(b),compute_metrics=metrics)
 trainer.train();result=trainer.evaluate();mlflow.log_metrics({k:float(v) for k,v in result.items()});trainer.save_model("transformer-model");tokenizer.save_pretrained("transformer-model")
