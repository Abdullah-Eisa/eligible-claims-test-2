# fastAPI

import io

from fastapi import FastAPI
''' from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel '''
import uvicorn

import pickle
import pandas as pd
''' import sklearn
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split '''
import xgboost as xgb
import json
# from onnxruntime import InferenceSession
import onnxruntime
from transformers import AutoTokenizer
import numpy as np
import scipy
from pathlib import Path
app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello , this is Eligible claims classifier app ,write (/docs) to go to app utilities"}

''' @app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id,"message": "Hello World"}



class Item(BaseModel):
    name: str
    price: float
    
    
@app.post("/items/")
def create_item(item: Item):
    return item '''
 

@app.post("/predict")
#def inference(representative_claim:str):
def inference(representative_claim:str,judge:str | None = None, 
            court:str | None = None):    
    # import onnx
    
    #base_path=""
    base_path=Path(__file__).resolve(strict=True).parent
    #print("base_path",base_path)

    tokenizers_files_path =str(base_path)+"/tokenizer_files"
    output_path = str(base_path)+"/setfitonnx_model.onnx"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizers_files_path)
    inputs = tokenizer(
        representative_claim,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="np",
    )
    
    inputs["input_ids"]=inputs["input_ids"].astype(np.int64)
    inputs["token_type_ids"]=inputs["token_type_ids"].astype(np.int64)
    inputs["attention_mask"]=inputs["attention_mask"].astype(np.int64)

    session = onnxruntime.InferenceSession(output_path)
    onnx_preds = session.run(None, dict(inputs))[0]



    # load dictionary
    with open(str(base_path)+"/judge_conf.pkl", 'rb') as fp:
        judge_conf = pickle.load(fp)

    with open(str(base_path)+"/eligible_per_judge.pkl", 'rb') as fp:
        eligible_per_judge = pickle.load(fp)

    with open(str(base_path)+"/court_conf.pkl", 'rb') as fp:
        court_conf = pickle.load(fp)
    
    with open(str(base_path)+"/eligible_per_court.pkl", 'rb') as fp:
        eligible_per_court = pickle.load(fp)

    judges=list(judge_conf.keys())
    
    courts=list(court_conf.keys())
    


    if judge==None or court==None or judge not in judges or court not in courts:
        return json.dumps({"succcess":True,"result":np.argmax(scipy.special.softmax(onnx_preds, axis=1)).item()})
    else:        
        model_input=pd.DataFrame([[judge_conf[judge],
                        court_conf[court],
                        eligible_per_judge[judge],
                        eligible_per_court[court],
                        scipy.special.softmax(onnx_preds, axis=1)[0][0]]
                        ],
                        columns=['judge_conf', 'court_conf', 'eligible_per_judge', 'eligible_per_court', 'FewShot'])        
      
      
        xgb_model = xgb.XGBClassifier(max_depth = 3, learning_rate=0.05,n_estimators=100,verposity=3,objective="binary:logistic", random_state=42)
        xgb_model.load_model(str(base_path)+"/xgb_model.json")  
        model_prediction=xgb_model.predict(model_input)
        return json.dumps({"succcess":True,"result":model_prediction.item()})
    
if __name__ == "__main__":
    uvicorn.run(app)

