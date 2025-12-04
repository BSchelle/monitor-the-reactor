<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# peux tu me génerer un canvas avec ce project mngmt pour que je puisse le sauvegarder et m'y referer ?

Voici un canvas simple (format texte) que tu peux copier/coller dans un doc / Notion / Miro et adapter.

## Vision / Objectif

Cas d’usage : diagnostic de défauts sur le Tennessee Eastman Process (classification normal vs fautes).
- Produit : démonstrateur complet avec EDA, modèles ML \& DL, API, interface Streamlit, packaging Docker, présentation PPT.
- Durée : 10 jours, équipe de 4 personnes (~40 j·H).


## Périmètre

- Données : dataset TEP existant (type Kaggle / Rieth) avec labels de fautes, sans modifier le simulateur.[^1][^2]
- Modèles :
    - Baseline ML : Random Forest / XGBoost sur features agrégées.
    - Modèle DL : UN seul modèle séquentiel (LSTM OU CNN 1D) sur fenêtres multivariées.
- App :
    - API (FastAPI ou équivalent) avec 1–2 endpoints de prédiction.
    - Front Streamlit simple (upload ou scénario préchargé, visualisation défauts dans le temps).
    - Docker : 2 services (API + Streamlit) + docker-compose.


## Équipe / Rôles

- Personne A : données, EDA, feature engineering, préprocessing.
- Personne B : modèles ML \& DL, évaluation.
- Personne C : API + packaging modèle, Docker backend.
- Personne D : Streamlit, UX démo, PPT, Docker frontend, coordination globale.


## Planning (10 jours)

- Jour 1
    - A : EDA rapide TEP, compréhension variables / segments.
    - B : définir tâche, fenêtres, splits.
    - C \& D : choix techno, création repo, structure projet.
- Jour 2
    - A : génération des fenêtres + labels, sauvegarde datasets prêts.
    - B : baseline ML (train/val/test), premières métriques.
    - C : squelette API (schémas et endpoints).
    - D : maquette UI Streamlit (layout).
- Jour 3
    - A : stabiliser pipeline préprocessing (scaler, encodage labels).
    - B : figer baseline ML + sauvegarde modèle.
    - C : brancher baseline dans l’API (`/predict`).
    - D : connecter Streamlit à l’API (appel baseline sur exemple).
- Jour 4
    - B : implémenter modèle DL (LSTM OU CNN 1D), premiers trainings.
    - A : support data (dataloaders, batching).
    - C : préparer endpoint `/predict_dl`.
    - D : ajouter plots temps + zones défauts.
- Jour 5
    - B : figer première version du modèle DL + sauvegarde.
    - A : script d’évaluation commun ML / DL (mêmes splits, métriques).
    - C : brancher DL dans l’API, tests unitaires.
    - D : UI Streamlit pour choisir modèle (ML vs DL).
- Jour 6
    - A \& B : petits ajustements hyperparamètres, vérif overfitting.
    - C : Dockerfile backend + test container API.
    - D : Dockerfile frontend + début docker-compose.
- Jour 7
    - Tous : finaliser docker-compose, tests end-to-end (front → API → modèles) sur scénario complet.
- Jour 8
    - A \& B : générer figures (confusion, courbes temps, importance variables).
    - C : logging minimal + endpoint `/health`.
    - D : polish UX Streamlit, messages d’erreur.
- Jour 9
    - D lead : PPT (contexte TEP, data, modèles, archi, démo, limites).[^2]
    - A/B/C : fournir schémas, graphiques, textes techniques.
- Jour 10
    - Tous : répétition de la démo de bout en bout, corrections finales (stabilité, temps de réponse, seeds).


## Livrables

- Code :
    - Module `preprocess + predict` partagé (ML \& DL).
    - Scripts d’entraînement / évaluation.
    - API (FastAPI) + tests de base.
    - App Streamlit.
- Infra :
    - 2 Dockerfiles (front, back) + `docker-compose.yml`.
- Data science :
    - Baseline ML + modèle DL avec métriques (accuracy, F1, matrices de confusion, courbes exemples).
- Business / démo :
    - PPT prêt à présenter.
    - Scénario de démo (fichier d’exemple, chemin utilisateur clair).

<div align="center">⁂</div>

[^1]: https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset

[^2]: https://download.arxiv.org/pdf/2208.01529v1.pdf


