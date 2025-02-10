FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

COPY models/random_forest_model.pkl models/scaler.pkl ./
CMD ["python", "src/app.py"]
