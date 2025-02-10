# Use uma imagem base do Python
FROM python:3.9-slim

# Defina o diretório de trabalho
WORKDIR /app

# Copie os arquivos necessários
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponha a porta que a API vai usar
EXPOSE 5000

# Comando para iniciar a API
COPY random_forest_model.pkl scaler.pkl ./
CMD ["python", "app.py"]
