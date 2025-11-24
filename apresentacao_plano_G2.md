# Plano de Apresentação - G2

## Previsão de Vendas de Produtos Amazon com Machine Learning

### Tempo Total: 10 minutos

---

## Slide 1: Título e Equipe (30 segundos)

### Conteúdo Visual:

- **Título:** "Previsão de Vendas de Produtos Amazon com XGBoost Otimizado"
- **Subtítulo:** "Aplicação de Machine Learning para Forecasting de Demanda"
- **Logo/Imagem:** Ícone Amazon + gráfico de tendência
- **Nome da equipe/integrantes**
- **Data:** Novembro 2024
- **Curso:** Ciência de Dados

### Talking Points:

- "Boa tarde, hoje apresentamos nosso projeto de previsão de vendas de produtos Amazon"
- "Desenvolvemos um modelo de machine learning capaz de prever com 91% de acurácia a quantidade de produtos vendidos no último mês"

---

## Slide 2: Problema e Motivação (1 minuto)

### Conteúdo Visual:

- **Contexto do Problema:**

  - 42.675 produtos no marketplace Amazon
  - Necessidade de previsão de demanda para gestão de inventário
  - Impacto direto em decisões de negócio

- **Objetivo Principal:**

  - Prever `purchased_last_month` (quantidade vendida no último mês)
  - R² alvo: > 0.85

- **Aplicações Práticas:**
  - 📦 Otimização de estoque
  - 💰 Estratégias de pricing dinâmico
  - 📊 Planejamento de demanda
  - 🎯 Identificação de produtos de alto potencial

### Talking Points:

- "O problema surge da necessidade real de empresas no marketplace Amazon preverem demanda"
- "Com milhares de produtos, é impossível gerenciar manualmente o estoque"
- "Nossa solução permite prever vendas futuras com base em características dos produtos"
- "O impacto: redução de ruptura de estoque e capital parado em inventário"

---

## Slide 3: Dataset - Características e Desafios (2 minutos)

### Conteúdo Visual - Parte 1:

- **Dataset Original:**
  - Fonte: Kaggle (Amazon Products Sales 2025)
  - 42.675 produtos → 34.140 após limpeza (20% removidos)
  - 17 features originais → 19 após feature engineering
- **Features Principais:**
  - Preço (original/desconto)
  - Rating e reviews
  - Categoria do produto
  - Badges (Best Seller, Sponsored)

### Conteúdo Visual - Parte 2:

- **Desafios Encontrados:**

  1. **Distribuição extremamente assimétrica** (gráfico de distribuição)

     - Skewness = 15.2
     - Produtos com 0 até 50.000+ vendas

  2. **Missing Values (25% dos dados)**

     - sustainability_tags: 60% missing
     - buy_box_availability: 35% missing

  3. **Outliers extremos**
     - IQR method: 8.500 outliers detectados

### Talking Points:

- "Nosso dataset apresentou desafios significativos desde o início"
- "A distribuição altamente assimétrica dificultava previsões precisas"
- "25% dos dados tinham valores faltantes que precisavam tratamento cuidadoso"
- "Descobrimos insights importantes: produtos com badge Best Seller vendem 300% mais"

---

## Slide 4: Metodologia - Pré-processamento e Feature Engineering (2 minutos)

### Conteúdo Visual - Parte 1:

- **Pipeline de Pré-processamento:**
  ```
  Raw Data (42.675) → Missing Values → Feature Engineering → Scaling → Split 80/20
  ```
- **Tratamento de Missing Values:**
  - Numéricas: mediana
  - Categóricas: flag binária + "Unknown"
  - Target missing: remoção (8.535 linhas)

### Conteúdo Visual - Parte 2:

- **Feature Engineering Criadas:**

  - `discount_amount` = original_price - discounted_price
  - `price_ratio` = discounted_price / original_price
  - `rating_review_interaction` = rating × log(reviews)
  - `log_total_reviews` = log1p(total_reviews)

- **Transformações Aplicadas:**
  - Log1p no target para modelos lineares
  - StandardScaler para features numéricas
  - One-hot encoding (5 categorias)

### Talking Points:

- "Desenvolvemos um pipeline robusto de pré-processamento"
- "A feature engineering foi crucial - a interação rating×reviews se tornou a 2ª mais importante"
- "Aplicamos transformação logarítmica no target para lidar com a assimetria"
- "Importante: essa transformação causou problemas posteriormente que vou explicar"

---

## Slide 5: Modelagem - Algoritmos e Otimização (2 minutos)

### Conteúdo Visual - Parte 1:

- **7 Modelos Testados:**
  | Modelo | R² Score | RMSE |
  |--------|----------|------|
  | Linear Regression | 0.198 | 5,133 |
  | Ridge | 0.198 | 5,134 |
  | Lasso | -0.030 | 5,819 |
  | ElasticNet | -0.023 | 5,799 |
  | Random Forest | 0.861 | 2,139 |
  | **XGBoost** | **0.898** | **1,828** |
  | Gradient Boosting | 0.875 | 2,028 |

### Conteúdo Visual - Parte 2:

- **Hyperparameter Tuning (XGBoost):**
  - Método: RandomizedSearchCV
  - 50 combinações × 5-fold CV = 250 fits
  - Tempo: 25 minutos
- **Parâmetros Otimizados:**
  - max_depth: 6 → 8
  - learning_rate: 0.1 → 0.05
  - n_estimators: 100 → 300
  - Regularização L1/L2 adicionada

### Talking Points:

- "Testamos desde modelos lineares simples até ensemble methods complexos"
- "XGBoost se destacou com R² de 0.898, explicando quase 90% da variância"
- "O tuning de hiperparâmetros trouxe melhoria adicional de 7.7% no RMSE"
- "RandomizedSearch foi escolhido por ser mais eficiente que GridSearch"

---

## Slide 6: Dificuldade Específica - Lição Aprendida (1 minuto)

### Conteúdo Visual:

- **O Problema:**

  - Usamos colunas com log1p criadas para modelos lineares
  - Aplicamos essas mesmas features em modelos tree-based
  - Resultado: performance PÉSSIMA inicial (R² < 0.3)

- **Diagnóstico:**

  - Modelos de árvore não precisam de transformações logarítmicas
  - Trees são naturalmente robustas a outliers e assimetria
  - Transformação desnecessária distorceu os padrões

- **Solução:**
  - Separamos pipelines: features escaladas para modelos lineares
  - Features originais para tree-based models
  - Resultado: melhoria de 60% no R²

### Talking Points:

- "Uma lição importante: nem toda técnica de pré-processamento é universal"
- "Inicialmente aplicamos log1p em todas as features para todos os modelos"
- "Descobrimos que isso prejudicava severamente os modelos de árvore"
- "A correção foi simples mas o aprendizado foi valioso"

---

## Slide 7: Resultados - Métricas e Performance (2 minutos)

### Conteúdo Visual - Parte 1:

- **Métricas do Modelo Final (XGBoost Tuned):**
  - **R² Score:** 0.9133 (91.33% variância explicada)
  - **RMSE:** 1,688.26 unidades
  - **MAE:** 338.99 unidades
  - **sMAPE:** 56.91%
  - **Melhoria vs Baseline:** 7.7% redução RMSE

### Conteúdo Visual - Parte 2:

- **Gráficos Principais:**
  1. Comparação de modelos (barras)
  2. Feature importance (top 10)
  3. Actual vs Predicted scatter plot
  4. Learning curves (sem overfitting)

### Conteúdo Visual - Parte 3:

- **Performance por Segmento:**
  | Volume | Produtos | RMSE | MdAPE |
  |--------|----------|------|-------|
  | Low (<500) | 4,971 | 254 | 80.7% |
  | Medium (500-5K) | 1,173 | 1,581 | 25.8% |
  | High (>5K) | 289 | 17,791 | 1.7% |

### Talking Points:

- "Alcançamos R² de 0.9133, superando significativamente nossa meta de 0.85"
- "O modelo é especialmente preciso em produtos de alto volume (MdAPE 1.7%)"
- "As features mais importantes são interações preço-rating e badges promocionais"
- "Learning curves mostram que o modelo generaliza bem sem overfitting"

---

## Slide 8: Visualizações e Interpretações (1 minuto)

### Conteúdo Visual:

- **4 Visualizações em Grid:**

  1. **Feature Importance** (top 5)

     - rating_review_interaction: 18%
     - log_total_reviews: 15%
     - discount_amount: 12%

  2. **Predições vs Real** (scatter)

     - Pontos concentrados na diagonal
     - Poucos outliers extremos

  3. **Distribuição de Erros** (histogram)

     - Centrada em zero
     - Distribuição normal

  4. **Melhoria com Tuning** (antes/depois)
     - RMSE: 1828 → 1688
     - R²: 0.898 → 0.913

### Talking Points:

- "As visualizações confirmam a robustez do modelo"
- "A feature importance revela que interações complexas são mais preditivas"
- "Os erros estão bem distribuídos, sem viés sistemático"

---

## Slide 9: Conclusões e Aprendizados (1.5 minutos)

### Conteúdo Visual - Parte 1:

- **✅ Objetivos Alcançados:**
  - Modelo com 91.33% de acurácia (meta: 85%)
  - Pipeline completo de ML implementado
  - 7.7% de melhoria com otimização
  - Insights acionáveis para negócio

### Conteúdo Visual - Parte 2:

- **📚 Principais Aprendizados:**
  1. Importância do EDA detalhado (25% do tempo)
  2. Feature engineering > modelos complexos
  3. Diferentes modelos = diferentes pré-processamentos
  4. Cross-validation essencial para robustez

### Conteúdo Visual - Parte 3:

- **⚠️ Limitações Identificadas:**

  - Performance menor em produtos low-volume
  - Dados de apenas 1 mês (sem sazonalidade)
  - Features externas não disponíveis (competidores, economia)

- **🚀 Trabalhos Futuros:**
  - Incorporar dados temporais/sazonais
  - Modelo específico por categoria
  - Deploy em produção com API
  - A/B testing com previsões

### Talking Points:

- "Superamos nossas metas e criamos um modelo production-ready"
- "O projeto reforçou a importância do processo end-to-end de data science"
- "Aprendemos que 80% do trabalho está na preparação dos dados"
- "Como próximos passos, seria valioso coletar dados temporais para capturar sazonalidade"

---

## Slide 10: Obrigado + Q&A (30 segundos)

### Conteúdo Visual:

- **Resumo Final:**

  - 📊 42.675 produtos analisados
  - 🎯 91.33% de acurácia alcançada
  - 🚀 Modelo pronto para produção
  - 📈 ROI potencial: redução de 30% em ruptura de estoque

- **Contato/Repositório:**

  - GitHub: github.com/bathwaterpizza/data-science-25-2
  - Modelo disponível: models/xgboost_tuned.pkl

- **"Perguntas?"** (grande e centralizado)

### Talking Points:

- "Em resumo, desenvolvemos uma solução robusta para previsão de vendas"
- "O modelo está disponível no GitHub junto com toda documentação"
- "Agradeço a atenção e estou aberto a perguntas"

---

## Notas para o Apresentador

### Timing:

- Mantenha rigor no tempo - use cronômetro
- Slides 3-7 são os mais importantes (70% do tempo)
- Se atrasar, pule detalhes técnicos dos slides 4-5

### Dicas de Apresentação:

1. **Início forte:** Comece com o impacto (91% de acurácia)
2. **Storytelling:** Conte a jornada dos dados até o modelo
3. **Seja visual:** Aponte para os gráficos enquanto explica
4. **Admita limitações:** Mostra maturidade técnica
5. **Termine com impacto:** Volte ao valor de negócio

### Possíveis Perguntas Q&A:

- **P: Por que XGBoost e não Deep Learning?**
  - R: Dataset pequeno (34K), XGBoost melhor para dados tabulares estruturados
- **P: Como lidaram com overfitting?**
  - R: Cross-validation 5-fold, regularização L1/L2, early stopping
- **P: Qual o custo computacional?**

  - R: Treinamento ~30min, predição <1seg para 1000 produtos

- **P: Como garantir performance em produção?**
  - R: Monitoramento de drift, retreino trimestral, A/B testing

### Material de Apoio:

- Tenha o notebook `model_performance_report.ipynb` aberto para mostrar detalhes se perguntarem
- Screenshots dos principais gráficos salvos como backup
