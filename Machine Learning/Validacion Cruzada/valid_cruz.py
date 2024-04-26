from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

#Cargamos un dataset de ejemplo

wine = datasets.load_wine()

X = wine.data
y = wine.target

#Creamos un clasificador Naive Bayes
clf = GaussianNB()

#Definimos las metricas a calcular

scoring = ['accuracy','precision_macro', 'recall_macro', 'f1_macro']

#Aplicar validación cruzada con multiples metricas
scores = cross_validate(clf, X, y, scoring=scoring, cv=10)

#Imprimir resultados
print("Accuracy: ", scores['test_accuracy'].mean())
print("Precision: ", scores['test_precision_macro'].mean())
print("Recall: ", scores['test_recall_macro'].mean())
print("F1: ", scores['test_f1_macro'].mean())

print(scores)