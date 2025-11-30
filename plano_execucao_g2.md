# Plano Final de Execução – Projeto de Ciência de Dados (G1 + G2)

**Tema:** Análise Preditiva de Vendas na Amazon (Eletrônicos)  
**Equipe:** Grupo 10 – 3WA

---

## 1. Definição do Problema e Objetivos

O projeto analisa como características de listagens de produtos eletrônicos da Amazon influenciam o volume de vendas (proxy: _purchased_last_month_) e constrói modelos preditivos capazes de estimar demanda.
Foram avaliados fatores como:

- Preço, desconto e competitividade
- Reputação (reviews e rating)
- Patrocínio (_is_sponsored_)
- Badges (Best Seller, cupons, ofertas)
- Indicadores derivados criados ao longo da G2

### Objetivos mensuráveis

- Construir modelos de regressão para prever demanda usando dados públicos.
- Comparar modelos lineares e modelos de árvore (RF/XGBoost).
- Avaliar desempenho via RMSE, MAE, R², sMAPE e MdAPE.
- Produzir pipeline reprodutível e notebooks finais consolidados.
- Criar apresentação final com insights e limitações.

---

## 2. Escopo e Requisitos (Atualizado na G2)

**Dataset:** Amazon Products Sales Dataset (~42 mil produtos).  
**Variáveis importantes:** preço original/atual, desconto %, reviews, rating, patrocinado ou não, badges, datas tratadas, e features derivadas (log1p, flags, outliers).  
**Limitações:** ausência de histórico temporal real, falta de categoria do produto e forte assimetria das métricas.  
**Ferramentas:** Google Colab, pandas, numpy, matplotlib, seaborn, sklearn, xgboost, SHAP.

---

## 3. Etapas do Projeto (Resumo Consolidado)

### G1 — Planejamento, Limpeza e EDA

- Limpeza profunda dos preços, reviews e ratings
- Normalização de booleans (+ criação de flags binárias)
- Transformações _log1p_
- Análise de outliers e comportamento da cauda
- Visualizações principais (distribuições, correlações, segmentações)

### G2 — Engenharia, Modelagem e Avaliação

- Baselines com Regressão Linear, Ridge, Lasso e ElasticNet
- Modelos tree-based: RandomForest, Gradient Boosting e XGBoost
- Hyperparameter tuning com RandomizedSearchCV (50 combinações, 5-fold CV)
- Validação cruzada, análise por segmentos de volume de vendas
- Interpretação com Feature Importance
- Consolidação total em notebook final + slides de apresentação

---

## 4. Cronograma Final (Sprints)

**Sprint 2 – EDA + Baseline**  
Limpeza final, flags, log1p, outliers, baseline linear.

**Sprint 3 – Modelos de Árvore**  
RandomForest/XGBoost, tuning leve e validação cruzada.

**Sprint 4 – Interpretabilidade**  
Feature importance, SHAP, erros por segmento.

**Sprint 5 – Consolidação**  
Notebook final, documentação, gráficos para apresentação, finalização dos slides.

---

## 5. Riscos e Mitigações

- **Cauda longa e assimetria:** mitigado com log1p e uso de métricas robustas.
- **Ruído em reviews/rating:** uso de flags e segmentações.
- **Reprodutibilidade:** criação de notebook final que roda de ponta a ponta.
- **Possível overfitting em árvores:** validação cruzada e controle de hiperparâmetros.

---

## 6. Entregáveis Finais (G1 + G2)

**Notebooks:**

- `exploratory_data_analysis.ipynb` — EDA inicial
- `phase1_preprocessing.ipynb` — Limpeza e feature engineering
- `phase2_model_training.ipynb` — Treinamento dos modelos
- `phase3_model_evaluation.ipynb` — Comparação inicial de modelos
- `phase4_hyperparameter_tuning.ipynb` — Otimização do XGBoost
- `phase5_improved_evaluation.ipynb` — Métricas avançadas
- `model_performance_report.ipynb` — Relatório final consolidado

**Modelos Treinados:** (`models/`)

- 7 modelos salvos em pickle (Linear, Ridge, Lasso, ElasticNet, RF, GB, XGBoost)
- `xgboost_tuned.pkl` — Modelo final otimizado
- `best_params.pkl` — Melhores hiperparâmetros

**Resultados e Visualizações:**

- CSVs com métricas: `model_comparison_results.csv`, `tuning_comparison.csv`, `stratified_performance.csv`
- Gráficos PNG: comparação de modelos, feature importance, learning curves, predições

**Documentação:**

- `README.md` — Descrição do projeto
- `plano_execucao_g2.md` — Este documento
- Slides finais da apresentação (PDF)

**Repositório:** https://github.com/bathwaterpizza/data-science-25-2

---

## 7. Critérios de Sucesso Atingidos

- Pipeline reprodutível testado do zero
- Comparação consistente entre 7 modelos diferentes
- Documentação clara e técnica
- Insights e limitações bem explicadas na apresentação
- Projeto pronto para avaliação final
- **Alvo inicial de R² >= 0.85 superado: R² = 0.9133 (91.33%)**

### Resultados do Modelo Final (XGBoost Tuned)

| Métrica  | Valor    |
| -------- | -------- |
| R² Score | 0.9133   |
| RMSE     | 1,688.26 |
| MAE      | 338.99   |
| sMAPE    | 56.91%   |
| MdAPE    | 62.18%   |

**Melhoria com hyperparameter tuning:** 7.7% de redução no RMSE em relação ao baseline.
