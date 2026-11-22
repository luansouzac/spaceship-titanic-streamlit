# Spaceship Titanic - Data App com Streamlit e Ngrok

Trabalho de **Visualização de Dados** (IF Sudeste MG). Aplicação web que prevê se um passageiro do
[Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) foi transportado para outra dimensão,
usando modelos de Machine Learning treinados por nós. O app roda no Google Colab e fica público por um túnel do Ngrok.

## O que tem aqui

| Arquivo / pasta | Para que serve |
|---|---|
| `spaceship_titanic.ipynb` | Notebook completo: análise exploratória, treino dos modelos, exportação dos `.pkl`, app Streamlit e túnel Ngrok |
| `app_spaceship.py` | Código do app (o notebook também cria este arquivo com `%%writefile`) |
| `modelos/` | Modelos treinados e exportados com `joblib` (`.pkl`), mais a lista de colunas e o padronizador |
| `data/` | `train.csv` e `test.csv` da competição no Kaggle |
| `.streamlit/config.toml` | Tema escuro e roxo do app |
| `prints_app/` | Capturas de tela do app funcionando |

## Como rodar no Google Colab (jeito pedido no trabalho)

1. Abra `spaceship_titanic.ipynb` no [Google Colab](https://colab.research.google.com).
2. Suba `train.csv` e `test.csv` (pasta `data/`) para `/content/`.
3. Crie uma conta em [dashboard.ngrok.com](https://dashboard.ngrok.com), copie o *Your Authtoken* e cole na célula do Ngrok (`NGROK_TOKEN`).
4. Execute todas as células (Ambiente de execução > Executar tudo). O notebook treina os modelos, salva os `.pkl` em `modelos/`, cria o app e imprime o **link público**.
5. Abra o link, preencha os dados do passageiro e clique em **Prever**.

Se o link der erro ou a página ficar em branco, rode a última célula (`!cat logs.txt`) para ver o erro.

## Como rodar no computador (opcional)

Com os `.pkl` já em `modelos/`:

```bash
pip install streamlit scikit-learn catboost joblib altair pandas
streamlit run app_spaceship.py
```

## O que o app faz

- **Prever:** o usuário preenche idade, planeta, destino, sono criogênico, VIP, gastos e detalhes da cabine. Ao clicar em *Prever*, o app monta as mesmas variáveis do treino, padroniza e mostra se foi transportado e a chance (%).
- **Gráficos:** taxa de transportados por sono criogênico, planeta, idade, deck e tamanho do grupo.
- **Barra lateral:** escolha entre CatBoost, Random Forest, KNN e Regressão Logística.

## Modelos

| Modelo | Acurácia (validação cruzada) |
|---|---|
| CatBoost | 0,812 |
| Random Forest | 0,804 |
| Regressão Logística | 0,788 |
| KNN | 0,784 |

O CatBoost é o melhor e fica como opção padrão. O pré-processamento (valores ausentes, variáveis novas, one-hot,
padronização) é o mesmo do trabalho anterior, e o app repete esse tratamento para o passageiro digitado.

## Grupo

- Luan Souza
- Francisco Carlos
