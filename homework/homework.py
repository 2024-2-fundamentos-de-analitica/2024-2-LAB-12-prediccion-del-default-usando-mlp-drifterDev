# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import gzip
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

def cargar_datos(ruta: str) -> pd.DataFrame:
    return pd.read_csv(ruta, index_col=False, compression='zip')

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={'default payment next month': 'default'})
    df = df.drop(columns=['ID'])
    df = df[df['MARRIAGE'] != 0]
    df = df[df['EDUCATION'] != 0]
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: x if x < 4 else 4)
    return df

def construir_pipeline() -> Pipeline:
    categorias = ["SEX", "EDUCATION", "MARRIAGE"]
    transformador = ColumnTransformer([
        ('one_hot', OneHotEncoder(dtype="int"), categorias),
    ], remainder=StandardScaler())
    
    return Pipeline([
        ('preprocesamiento', transformador),
        ('reduccion_dimensionalidad', PCA(n_components=0.95)),
        ('seleccion_atributos', SelectKBest(score_func=f_regression)),
        ('modelo', MLPClassifier(max_iter=30000))
    ])

def optimizar_modelo(pipeline: Pipeline, x: pd.DataFrame) -> GridSearchCV:
    parametros = {
        "seleccion_atributos__k": range(1, len(x.columns) + 1),
        "modelo__hidden_layer_sizes": [(n,) for n in range(1, 11)],
        "modelo__solver": ["adam"],
        "modelo__learning_rate_init": [0.01, 0.001, 0.0001],
    }
    return GridSearchCV(pipeline, param_grid=parametros, cv=10, scoring='balanced_accuracy', refit=True, verbose=2)

def guardar_modelo(ruta: str, modelo: GridSearchCV):
    with gzip.open(ruta, 'wb') as archivo:
        pickle.dump(modelo, archivo)

def calcular_metricas(nombre: str, y_real, y_predicho) -> dict:
    return {
        'tipo': 'metricas',
        'conjunto': nombre,
        'precision': precision_score(y_real, y_predicho),
        'precision_balanceada': balanced_accuracy_score(y_real, y_predicho),
        'recall': recall_score(y_real, y_predicho),
        'f1_score': f1_score(y_real, y_predicho)
    }

def calcular_matriz_confusion(nombre: str, y_real, y_predicho) -> dict:
    cm = confusion_matrix(y_real, y_predicho)
    return {
        'tipo': 'matriz_confusion',
        'conjunto': nombre,
        'true_0': {"predicho_0": int(cm[0][0]), "predicho_1": int(cm[0][1])},
        'true_1': {"predicho_0": int(cm[1][0]), "predicho_1": int(cm[1][1])}
    }

def main():
    ruta_datos = 'files/input/'
    ruta_modelos = 'files/models/'
    
    test_df = cargar_datos(os.path.join(ruta_datos, 'test_data.csv.zip'))
    train_df = cargar_datos(os.path.join(ruta_datos, 'train_data.csv.zip'))
    
    test_df = limpiar_datos(test_df)
    train_df = limpiar_datos(train_df)
    
    x_test, y_test = test_df.drop(columns=['default']), test_df['default']
    x_train, y_train = train_df.drop(columns=['default']), train_df['default']
    
    pipeline = construir_pipeline()
    modelo = optimizar_modelo(pipeline, x_train)
    modelo.fit(x_train, y_train)
    
    guardar_modelo(os.path.join(ruta_modelos, 'modelo.pkl.gz'), modelo)
    
    y_test_pred = modelo.predict(x_test)
    y_train_pred = modelo.predict(x_train)
    
    metricas_test = calcular_metricas('test', y_test, y_test_pred)
    metricas_train = calcular_metricas('train', y_train, y_train_pred)
    
    matriz_conf_test = calcular_matriz_confusion('test', y_test, y_test_pred)
    matriz_conf_train = calcular_matriz_confusion('train', y_train, y_train_pred)
    
    with open('files/output/metricas.json', 'w') as archivo:
        archivo.write(json.dumps(metricas_train) + '\n')
        archivo.write(json.dumps(metricas_test) + '\n')
        archivo.write(json.dumps(matriz_conf_train) + '\n')
        archivo.write(json.dumps(matriz_conf_test) + '\n')

if __name__ == "__main__":
    main()
