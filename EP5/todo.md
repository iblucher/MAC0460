# TODO EP5

- Dataset do MNIST (ver exemplo EP4)
    - [X] Selecionar classes 0 a 4
    - [X] Montar conjunto de teste
    - [X] Montar conjunto de treinamento + validação
        - [X] 500 exemplos **aleatórios** de cada classe
        - [X] Formato flattened
        - [X] Valores normalizados no intervalo [0, 1]
    - [X] Montar arrays de rótulos para treino e teste
    - [X] Embaralhar X_train e Y_train

- Cross-validation (5-fold)
    - [X] SVM
    - [X] Rede neural
    - [X] Cross-validation accuracy para selecionar o melhor algoritmo

- Teste final
    - Escolher algoritmo e parâmetros que performaram melhor e treinar com todos os dados
    - Calcular acurácia com respeito ao conjunto de teste

- Relatório
    - Implementação
    - Shape de X_train e Y_train
    - Shape de X_test e Y_test
    - Valores mínimos e máximos de X_train, Y_train, X_test e Y_test (?)
    - Número de exemplos no fold e acurácia no cross-validation para cada algoritmo
    - Cross-validation accuracy (média das acurácias por fold)
    - Características do conjunto de teste e desempenho do algoritmo escolhido com respeito ao conjunto de teste (acurácia e matriz de confusão)