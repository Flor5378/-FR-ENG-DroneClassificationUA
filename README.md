#  Military Drone Classification Project – Ukraine Context

##  Context
In the context of modern warfare, the use of military drones in Ukraine has become a major strategic factor. This project focuses on applying **Machine Learning** to classify various types of drones—whether used for **reconnaissance** or as **weapons of destruction**. The ability to differentiate drones in **real time** enhances the effectiveness of countermeasures against enemy strategies.

##  Project Objectives
The goal of this project is to use machine learning techniques—particularly the **Random Forest** algorithm—to classify drone trajectories based on their type and purpose. The classifier aims to determine whether a drone is a reconnaissance tool or an active threat.

##  Project Workflow

### 1. **OSINT Phase** (Open Source Intelligence)
- Collect geographic coordinates of Russian drone bases.
- Gather technical specifications of Russian and Ukrainian drone models.

### 2. **Data Generation**
- Simulate a dataset of **1,000 randomly generated drone trajectories**.

### 3. **Machine Learning Classification**
- Train a **Random Forest** model to classify drone trajectories based on their features and origin.

### 4. **Results & Evaluation**
- Compare the model's predictions with real-world labels to evaluate performance.

##  Project Files

| File Name | Description |
|-----------|-------------|
| `1 - generationTrajectoires - JAFFUEL SIMON.ipynb` | Jupyter notebook for drone trajectory generation |
| `2 - classificationTrajectoires - JAFFUEL SIMON.ipynb` | Jupyter notebook for model training and classification |
| `bases_lancement.csv` | Data on drone launch bases |
| `caracteristique_drone.csv` | Specs of Russian and Ukrainian drones |
| `DroneProject.zip` | Compressed archive containing the full project |
| `Présentation projet Ukraine.pdf` | Slide deck summarizing project results |

##  Tech Stack
- **Python**
- **Pandas / NumPy**
- **Scikit-learn**
- **Jupyter Notebook**

## Results Overview (v2)

![v2 summary: simulated radar tracks, test scores, confusion matrix, model comparison and early classification](v2/figures/40_synthese_v2.png)

Details, notebooks and interactive maps: [`v2/README.md`](v2/README.md).

## ✅ Conclusion
This project demonstrates how **AI** and **Machine Learning** are reshaping modern conflict analysis. Accurate classification of drones can lead to **faster response times** and **enhanced threat detection**, offering a technological edge on the battlefield.

## 👥 Authors
- **JAFFUEL Romain**  
- **SMITH William**  
- **GROLLEAU Florian**
