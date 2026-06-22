from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def home():
    return {"message" : "Welcome to fastAPI Series!"}
@app.get("/contact")
def contact():
    return {"message" : "You can connect us any time."}