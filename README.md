# Projet 7 Openclassrooms Datascientist 

Ce projet a pour but de déployer un modèle via une API (ici Flask) dans le Web en utilisant un dashboard (Streamlit) pour présenter le travail de modélisation. Il comporte aussi un test unitaire à l'aide de Pytest.
Le code source de l'API contient entre autres un calculateur de score pour une demande de crédit bancaire qui retourne au Dashboard la probabilité que le client puisse le rembourser, et qui indique donc si le crédit est accordé ou non.


## Pré-requis

* Installer Python 3 : [Téléchargement Python 3](https://www.python.org/downloads/)
* Installer git : [Téléchargement Git](https://git-scm.com/book/fr/v2/D%C3%A9marrage-rapide-Installation-de-Git)

## Installation

### 1. Télécharger le projet sur votre répertoire local

### 2. Mettre en place un environnement virtuel :
* Créer l'environnement virtuel: `python -m venv venv`
* Activer l'environnement virtuel :
    * Windows : `venv\Scripts\activate.bat`
    * Unix/MacOS : `source venv/bin/activate`

    
### 3. Dépendances du projet :
```
pip install -r requirements.txt
```

### 4. Découpage des dossiers :
API.py : fichier python de l'API (Flask)
Dashboard.py : fichier python du Dashboard (Streamlit)
DATA : dossier data 	- Autre : Sous-dossier1 avec fichiers liés au modèle
            			- Source : Sous-dossier2 avec fichier data source qui a déjà subi le même préprocessing que le fichier d'entraînement
MODELS : dossier contenant le modèle de prédiction XGBoost, l'imputer (KNN) et le standardScaler
tests : dossier de test unitaire avec Pytest
Logo.png : logo de la société

## Démarrage
* Lancer le script à l'aide de la commande suivante : `streamlit run Dashboard.py`



