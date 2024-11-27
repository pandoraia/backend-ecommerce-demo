# Usar una imagen base de Python 3.12
FROM python:3.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de tu proyecto
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de tu código
COPY ./app ./app

# Comando para ejecutar tu aplicación
CMD uvicorn app.main:app --port=8000 --host=0.0.0.0