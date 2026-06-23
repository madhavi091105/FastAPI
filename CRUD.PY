from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
app = FastAPI()
class Patient(BaseModel):
    id:int
    name:str
    age:int
    city:str
patients = [
    { 
     "id":1,
     "name":"Rahul",
     "age":25,
     "city":"Delhi"
     }
]
@app.get("/patients/{patient_id}")
def get_patient(patient_id: int):
    for patient in patients:
        if patient["id"] == patient_id:
            return patient

    raise HTTPException(
        status_code=404,
        detail="Patient not found"
    )


# POST (CREATE NEW PATIENT)
@app.post("/patients")
def create_patient(patient: Patient):

    patients.append(patient.model_dump())

    return {
        "message": "Patient added successfully",
        "patient": patient
    }


# PUT (UPDATE PATIENT)
@app.put("/patients/{patient_id}")
def update_patient(patient_id: int, updated_patient: Patient):

    for index, patient in enumerate(patients):

        if patient["id"] == patient_id:

            patients[index] = updated_patient.model_dump()

            return {
                "message": "Patient updated successfully",
                "patient": updated_patient
            }

    raise HTTPException(
        status_code=404,
        detail="Patient not found"
    )


# DELETE PATIENT
@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: int):

    for index, patient in enumerate(patients):

        if patient["id"] == patient_id:

            deleted_patient = patients.pop(index)

            return {
                "message": "Patient deleted successfully",
                "patient": deleted_patient
            }

    raise HTTPException(
        status_code=404,
        detail="Patient not found"
    )