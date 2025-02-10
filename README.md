# 📊 Tax Fraud Detection

Este repositório contém um projeto de aprendizado de máquina para a detecção de inconsistências em impostos, utilizando Random Forest e processamento de dados com Pandas e Scikit-Learn.

## 🚀 Tecnologias Utilizadas
- Python
- Flask
- Pandas
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- Seaborn & Matplotlib
- Joblib
- Docker (opcional)

## 📂 Estrutura do Projeto
```
.
├── data
│   ├── dataset_to_candidate.csv  # Base de dados
│
├── figures  # Visualizações geradas pela EDA
│   ├── distribuicao_classes.png
│   ├── histogramas_impostos.png
│   ├── importancia_features.png
│   ├── mapa_calor_correlacao.png
│   ├── matriz_confusao.png
│
├── models  # Modelos treinados
│   ├── random_forest_model.pkl
│   ├── scaler.pkl
│
├── src  # Código-fonte
│   ├── app.py  # API Flask para previsões
│   ├── eda.py  # Análise exploratória e treinamento do modelo
│
├── venv  # Ambiente virtual (ignorar no Git)
├── dockerfile  # Configuração Docker
├── requirements.txt  # Dependências do projeto
├── README.md  # Documentação
```

## 📊 Análise Exploratória & Treinamento
Para executar a análise exploratória e treinar o modelo, rode:
```bash
python src/eda.py
```
Isso gera visualizações e treina um modelo Random Forest salvo em `models/`.

## 🔥 Executando a API Flask
Para rodar a API de previsão:
```bash
python src/app.py
```
Ela iniciará na porta `5000`. Para testar uma previsão, envie um POST para `http://localhost:5000/predict` com um JSON:
```json
{
  "features": [valor1, valor2, valor3, ...]
}
```

## 🐳 Rodando com Docker (Opcional)
```bash
docker build -t tax-fraud-detection .
docker run -p 5000:5000 tax-fraud-detection
```

## 📌 Instalação das Dependências
Crie um ambiente virtual e instale as dependências:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```
