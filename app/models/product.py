from pydantic import BaseModel, Field
from typing import Optional
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError('Invalid objectid')
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, _model):
        # En Pydantic v2, se usa este método para modificar el esquema JSON
        schema.update(type="string")
        return schema

# Definimos el modelo de Producto
class Product(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id")  # Para MongoDB, el campo _id es tratado como ObjectId
    name: str

    class Config:
        # Cambiado de allow_population_by_field_name a populate_by_name
        populate_by_name = True
        json_encoders = {ObjectId: str}  # Cuando devuelves _id, lo devuelve como un string en JSON
