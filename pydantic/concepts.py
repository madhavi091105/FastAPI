from pydantic import BaseModel, Field, field_validator, model_validator, computed_field
from typing import List
class Address(BaseModel):
    city: str
    state: str
    pincode: str
class Patient(BaseModel):
    id: int
    name: str
    age: int = Field(gt=0)
    weight: float
    height: float
    address: Address
    diseases: List[str]
    @field_validator("name")
    @classmethod
    def validate_name(cls,value):
        if len(value) < 3:
            raise ValueError("Name must have at least 3 characters")
        return value.title()
    @model_validator(mode="after")
    def validate_age_weight(self):
        if self.age < 18 and self.weight > 100:
            raise ValueError(
                "Weight seems unrealistic"
            )
        return self
    @computed_field
    @property
    def bmi(self) -> float:
        return round(self.weight / ((self.height / 100) ** 2), 2)
patient = Patient(
    id=1,
    name="madhavi",
    age=21,
    weight=55,
    height=160,
    address={
        "city": "Ropar",
        "state": "Punjab",
        "pincode": "140001"
    },
    diseases=["Fever", "Cold"]
)

print(patient)
print(patient.bmi)

# Serialization
print("\nDictionary:")
print(patient.model_dump())

print("\nJSON:")
print(patient.model_dump_json(indent=4))