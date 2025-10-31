from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello Sahana, your AI English Tutor backend is working!"}
