import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class BiharVoter(BaseModel):
    Age_Group: str
    Gender: str
    Geography: str
    Education: str
    Occupation: str
    Caste: str

bihar_model = joblib.load("models/Bihar_voter_prediction.pkl")

@app.get("/")
async def root():
    return {"status": "active", "models": ["bihar"]}

@app.post("/bihar")
async def predict_bihar(data: BiharVoter):
    df = pd.DataFrame([data.model_dump()])
    prediction = bihar_model.predict(df)[0]
    return {"voted_party": prediction}