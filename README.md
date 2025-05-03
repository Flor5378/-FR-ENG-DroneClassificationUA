# Projet Drones Militaires en Ukraine

## Contexte
Dans un contexte de guerre moderne, l’utilisation de drones militaires en Ukraine est devenue un enjeu majeur. Ce projet se concentre sur l'application du **Machine Learning** pour classifier différents types de drones, qu'ils soient utilisés pour des missions de reconnaissance ou comme armes de destruction. La capacité à différencier ces drones en temps réel permet de mieux contrer les stratégies militaires ennemies.

## Objectifs du projet
Le projet vise à utiliser des techniques de machine learning, notamment l'algorithme **Random Forest**, pour classifier des trajectoires de drones en fonction de leur type et usage. Le but est de déterminer si un drone est destiné à des missions de reconnaissance ou s’il est une menace en tant qu'arme de destruction.

## Étapes du projet

1. **Phase OSINT** (Open Source Intelligence) :
   - Récupération des coordonnées des bases russes.
   - Collecte des caractéristiques des différents modèles de drones russes et ukrainiens.
   
2. **Génération des données** :
   - Création d'un jeu de données avec 1000 trajectoires de drones générées aléatoirement.

3. **Classification avec Machine Learning** :
   - Application de l'algorithme **Random Forest** pour classifier les trajectoires de drones en fonction de leurs caractéristiques et de leur origine.
   
4. **Résultats** :
   - Comparaison des résultats obtenus avec les données réelles et prédictions de l'algorithme.

## Fichiers du projet

Voici la liste des fichiers inclus dans ce projet :

- `1 - generationTrajectoires - JAFFUEL SIMON.ipynb` : Script Jupyter pour la génération des trajectoires de drones.
- `2 - classificationTrajectoires - JAFFUEL SIMON.ipynb` : Script Jupyter pour l'application du modèle de machine learning.
- `bases_lancement.csv` : Données sur les bases de lancement de drones.
- `caracteristique_drone.csv` : Caractéristiques des drones russes et ukrainiens.
- `DroneProject.zip` : Fichier compressé permettant de refaire le projet
- `Présentation projet Ukraine.pdf` : Présentation des résultats du projet.

## Conclusion
Ce projet s'inscrit dans une évolution technologique rapide, où l'**IA** et le **Machine Learning** modifient la donne dans les conflits modernes. Les résultats obtenus ouvrent la voie à une meilleure identification des menaces potentielles et à une prise de décision plus rapide dans des situations de guerre.

## Auteur
- **JAFFUEL Romain**
