import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Definindo caminhos
base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path, "..", "data", "dataset_to_candidate.csv")
figures_path = os.path.join(base_path, "..", "figures")
models_path = os.path.join(base_path, "..", "models")

# Criando diretórios caso não existam
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# Carregando os dados
df = pd.read_csv(data_path, sep=';')
print(df.head())
df.info()
print(df.describe())

print("\nValores nulos por coluna:\n", df.isnull().sum())

df.fillna(df.median(numeric_only=True), inplace=True)

df['issue_date'] = pd.to_datetime(df['issue_date'], errors='coerce')

# Criando novas features
df['year'] = df['issue_date'].dt.year
df['month'] = df['issue_date'].dt.month

# Convertendo colunas categóricas para numéricas
le = LabelEncoder()
df['state'] = le.fit_transform(df['state'])
df['class_label'] = df['class_label'].map({'valid': 1, 'not valid': 0})

# Selecionando features e target
X = df.drop(columns=['issue_date', 'class_label'])
y = df['class_label']

# Dividindo o dataset em treino (70%), validação (15%) e teste (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Aplicando SMOTE para balancear as classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nDistribuição das classes após SMOTE (Apenas Treinamento):\n", pd.Series(y_train_resampled).value_counts())

# Normalizando os dados numéricos
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Treinando um modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Acurácia do modelo: {accuracy:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_val, y_pred))

# Salvando figuras na pasta figures
conf_matrix = confusion_matrix(y_val, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.savefig(os.path.join(figures_path, "matriz_confusao.png"))
plt.show()

sns.countplot(x=y_train_resampled)
plt.title("Distribuição das Classes Após SMOTE (Apenas Dados de Treino)")
plt.savefig(os.path.join(figures_path, "distribuicao_classes.png"))
plt.show()

importances = model.feature_importances_
features = X.columns  # nomes das colunas originais sem 'issue_date' e 'class_label'

feat_importances = pd.DataFrame({'feature': features, 'importance': importances})
feat_importances = feat_importances.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feat_importances)
plt.title('Importância das Variáveis - Random Forest')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "importancia_features.png"))
plt.show()

# Criando histogramas das principais variáveis de impostos
impostos = ['calculated_value', 'iss_tax_rate', 'inss_tax_rate', 'csll_tax_rate', 
            'ir_tax_rate', 'cofins_tax_rate', 'pis_tax_rate']

plt.figure(figsize=(12, 8))
for i, coluna in enumerate(impostos, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[coluna], bins=30, kde=True)
    plt.title(f'Distribuição de {coluna}')
    
plt.tight_layout()
plt.savefig(os.path.join(figures_path, "histogramas_impostos.png"))
plt.show()

# Gerando a matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Mapa de Calor - Correlação entre as Variáveis")
plt.savefig(os.path.join(figures_path, "mapa_calor_correlacao.png"))
plt.show()

# Salvando o modelo e o scaler na pasta models
joblib.dump(model, os.path.join(models_path, 'random_forest_model.pkl'))
joblib.dump(scaler, os.path.join(models_path, 'scaler.pkl'))

print("Colunas usadas no treino:", X_train.columns.tolist())
print("Número de features usadas:", X_train.shape[1])
