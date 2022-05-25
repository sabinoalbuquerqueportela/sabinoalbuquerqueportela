# **Caracterização de Asteróides com o uso de IA**


# **Importando as Bibliotecas**

# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# **Parametros do Site**
# load through url
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset = pd.read_csv(url, names = attributes)
dataset.columns = attributes
# **Paramentros do Nosso arquivo**
# load through url
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset = pd.read_csv(url, names = attributes)
dataset.columns = attributes
# **Informações das Tabelas**
# shape
print(dataset.shape)

# head
print(dataset.head(20))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())

# **Gerando os Gráficos**

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()
# Aplicação da **IA**

# **5. Avalie Algoritmos**
Agora é hora de criar alguns modelos dos dados e estimar sua precisão em dados não vistos.

Aqui está o que vamos cobrir nesta etapa:

Separe um conjunto de dados de validação.
Configure os testes para utilizar validação cruzada de 10.
Construa 5 modelos diferentes para prever espécies a partir de medições de flores
Selecione o melhor modelo.

**5.1 Criar um conjunto de dados de validação**

Precisamos saber que o modelo que criamos é bom.

Mais tarde, usaremos métodos estatísticos para estimar a precisão dos modelos que criamos em dados não vistos. Também queremos uma estimativa mais concreta da precisão do melhor modelo em dados não vistos, avaliando-o em dados reais não vistos.

Ou seja, vamos reter alguns dados que os algoritmos não conseguirão ver e usaremos esses dados para obter uma segunda e independente ideia de quão preciso o melhor modelo pode realmente ser.

Vamos dividir o conjunto de dados carregado em dois, 80% dos quais usaremos para treinar nossos modelos e 20% que iremos reter como um conjunto de dados de validação.
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
Agora você tem dados de treinamento no X_train e no Y_train para preparar modelos e um X_validation e Y_validation sets que podemos usar mais tarde.

Observe que usamos uma fatia python para selecionar as colunas na matriz NumPy. Se isso é novidade para você, talvez você queira fazer o check-out desta postagem:

https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/

**5.2 Test Harness**
 
Usaremos a validação cruzada (cross-validation) de 10 vezes para estimar a precisão.

Isso dividirá nosso conjunto de dados em 10 partes, treinar em 9 e testar em 1 e repetir para todas as combinações de divisões de teste de trem.
# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
A semente aleatória (random seed) específica não importa, aprenda mais sobre geradores de números pseudo-aleatórios aqui:

https://machinelearningmastery.com/introduction-to-random-number-generators-for-machine-learning/

Estamos usando a métrica de “accuracy” para avaliar modelos. Essa é uma proporção do número de instâncias corretamente previstas divididas pelo número total de instâncias no conjunto de dados multiplicado por 100 para fornecer uma porcentagem (por exemplo, 95% de accurate). Nós estaremos usando a variável de pontuação quando nós executarmos construir e avaliar cada modelo a seguir.

# **5.3 Construir Modelos**
Não sabemos quais algoritmos seriam bons nesse problema ou quais configurações usar. Nós temos uma ideia das parcelas de que algumas das classes são parcialmente linearmente separáveis ​​em algumas dimensões, então esperamos resultados geralmente bons.

Vamos avaliar 6 algoritmos diferentes:

Regressão Logística (LR)
Análise Linear Discriminante (LDA)
K-vizinhos mais próximos (KNN).
Árvores de Classificação (Decision Tree) e Regressão (CART).
Gaussian Naive Bayes (NB).
Support Vector Machines (SVM).
Esta é uma boa mistura de algoritmos simples lineares (LR e LDA), não lineares (KNN, CART, NB e SVM). Redefinimos a semente numérica aleatória antes de cada execução para garantir que a avaliação de cada algoritmo seja executada usando exatamente as mesmas divisões de dados. Isso garante que os resultados sejam diretamente comparáveis.

Vamos construir e avaliar nossos modelos:

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
   kfold = model_selection.KFold(n_splits=10)
   cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
   results.append(cv_results)
   names.append(name)
   msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
   print(msg)
5.4 Selecione o melhor modelo**

Agora temos 6 modelos e estimativas de precisão para cada um. Precisamos comparar os modelos entre si e selecionar os mais precisos.

Executando o exemplo acima, obtemos os seguintes resultados brutos:

Note que os seus resultados podem ser diferentes. Para mais informações, veja o post:

https://machinelearningmastery.com/randomness-in-machine-learning/

Também podemos criar um gráfico dos resultados da avaliação do modelo e comparar o spread e a accuracy média de cada modelo. Há uma população de medidas de accuracy para cada algoritmo porque cada algoritmo foi avaliado 10 vezes (10 fold cross validation).


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
Você pode ver que as plotagens de caixa e bigode estão esmagadas no topo da faixa, com muitas amostras atingindo 100% de accuracy.

# **6. Faça previsões**
O algoritmo KNN é muito simples e foi um modelo preciso baseado em nossos testes. Agora queremos ter uma ideia da accuracy do modelo em nosso conjunto de validação.

Isso nos dará uma verificação final independente da accuracy do melhor modelo. É importante manter um conjunto de validação apenas no caso de você ter feito uma falha durante o treinamento, como overfitting no conjunto de treinamento ou um vazamento de dados. Ambos resultarão em um resultado excessivamente otimista.

Podemos executar o modelo KNN diretamente no conjunto de validação e resumir os resultados como uma pontuação final de accuracy , uma matriz de confusão e um relatório de classificação.

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
Podemos ver que a accuracy é de 0,9 ou 90%. A matriz de confusão fornece uma indicação dos três erros cometidos. Finalmente, o relatório de classificação fornece um detalhamento de cada classe por precisão, recall, pontuação-f1 e suporte mostrando resultados excelentes (dado que o conjunto de dados de validação era pequeno).

Você pode aprender mais sobre como fazer previsões e prever probabilidades aqui:

https://machinelearningmastery.com/make-predictions-scikit-learn/
