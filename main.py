from fastapi import FastAPI
import json
app = FastAPI()
@app.get("/")
def load_data():
    with open('patients.json','r') as f:
        data = json.load(f)
    return data
        
        
def hello():
    return {"message" : "Patient Management System API"}
@app.get("/contact")
def about():
    return {"message" : "A fully functional API to manage your patient record."}
@app.get('/view')
def view():
    data = load_data()
    return data