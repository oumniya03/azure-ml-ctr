#  🚀 Prédiction CTR Multimodale sur MicroLens-1M 
<img width="800" height="400" alt="ctr" src="https://github.com/user-attachments/assets/bc454778-a3fe-4029-8454-3b196c4d46e6" />

## 📌 Aperçu du Projet

Ce dépôt contient la solution pour la **Compétition de Prédiction du Taux de Clics Multimodaux (CTR)** basée sur le jeu de données 
[MicroLens-1M](https://recsys.westlake.edu.cn/MicroLens_1M_MMCTR/)

**Objectif:** Prédire la probabilité qu'un utilisateur clique sur un élément spécifique (vidéo/article) en se basant sur :
- **Historique Utilisateur :**  Comportement séquentiel (clics passés).
- **Contenu de l'Élément :** Caractéristiques multimodales (Titres Textuels + Couvertures Images).

## 🎯 Approche Stratégique
Mon approche repose sur une **architecture d'Optimisation en Cascade.** 
Contrairement à une méthode 'end-to-end' souvent trop lourde, j’ai séparé l'extraction de connaissances (Task 1) de la modélisation comportementale (Task 2). Cette stratégie m’a permis d'utiliser des modèles de pointe comme CLIP tout en gardant un modèle de prédiction agile.

- **Task 1** : Extraction d'embeddings multimodaux avec CLIP
- **Task 2** : Modèle CTR avec architecture Attention + DNN
- **Meilleur AUC** : 0.7752 (validation)

## 🏗️ Architecture
**Pipeline en 3 Étapes**

### 1.Extraction Multimodale (CLIP)

- **Modèle :** openai/clip-vit-base-patch32
- **Fusion :** Embeddings Texte (512D) + Image (512D) = 1024D

### 2.Réduction Dimensionnelle (PCA)

- **Compression :** 1024D → 128D
- **Normalisation L2** pour stabilité

### 3.Modèle CTR (Attention + DNN)
- **User/Item** Embeddings (64D)
- **Multi-head Attention** sur historique
- **Deep Neural Network** [512→256→128→1]

<img width="1261" height="1364" alt="diagram-export-20-12-2025-12_45_52" src="https://github.com/user-attachments/assets/ca8ea7d4-4550-460c-906a-15632be4201f" />

## 📸 Explication détaillé

> 🎥 **[Voir la vidéo de l'explication sur YouTube](https://youtu.be/VuLjZuhNcho?si=-xPEZ3COH0QGKKVa)**

## 📦 Installation
```bash
# Cloner le dépôt
git clone https://github.com/oumniya03/Projet_competition.git
cd Projet_competition

# Installer les dépendances
pip install torch torchvision transformers pandas numpy scikit-learn tqdm
```

**Prérequis** :
- Python 3.11+
- CUDA 11.8+ (recommandé pour GPU)
  
## 🚀 Utilisation
```bash
# Exécuter le notebook complet
jupyter notebook competition.ipynb
```

## 📊 Résultats
- AUC validation : **0.7752** (GPU, Kaggle Tesla T4)
- AUC sur Azure ML : **~0.51** (CPU Standard_DS3_v2, run limité — voir explication ci-dessous)
- Dataset : 3.6M interactions train, 91K items

## 🛠️ Stack Technique
- Deep Learning : PyTorch 2.0, Transformers (CLIP)
- Data Processing : Pandas, NumPy
- ML Tools : scikit-learn (PCA), tqdm
- Compute : NVIDIA Tesla T4 (Kaggle) / Azure Standard_DS3_v2

---

## ☁️ MLOps sur Azure ML

### Pipeline end-to-end

```
[Données Parquet]          [GitHub Push]
       │                        │
       ▼                        ▼
[Azure Blob Storage]    [GitHub Actions CI/CD]
       │                        │
       └──────────┬─────────────┘
                  ▼
         [Azure ML Workspace]
                  │
         ┌────────┴────────┐
         ▼                 ▼
   [Training Job]    [Model Registry]
   src/train.py      ctr-multimodal v1
   MLflow tracking         │
                           ▼
                  [Batch Endpoint]
                  ctr-batch-endpoint
                  batch_score.py
                           │
                           ▼
                  [monitor.py]
                  PSI + KS drift detection
```

### Composants déployés

| Composant | Statut | Détail |
|---|---|---|
| Workspace Azure ML | ✅ | `ctr-workspace`, `rg-ctr-project`, `francecentral` |
| Compute cluster | ✅ | `gpu-cluster`, Standard_DS3_v2, min 0 / max 1 |
| Données | ✅ | `microlens-data`, 4 fichiers parquet |
| Entraînement + MLflow | ✅ | `src/train.py`, early stopping, tracking complet |
| Registre modèle | ✅ | `ctr-multimodal v1`, CUSTOM_MODEL |
| CI/CD GitHub Actions | ✅ | `.github/workflows/deploy.yml`, mode `deploy-only` / `retrain` |
| Batch Endpoint | ✅ déployé | `ctr-batch-endpoint`, modèle charge correctement |
| Online Endpoint | ❌ abandonné | Voir ci-dessous |
| Monitoring | 🔧 code prêt | `monitor.py`, PSI + KS, non testé en prod |

### Choix du Batch Endpoint

Les **Managed Online Endpoints** Azure ML ont été abandonnés après confirmation d'une limitation de souscription (`SubscriptionNotRegistered` persistant malgré l'enregistrement de tous les resource providers nécessaires). Cette erreur est liée au type de souscription personnelle (`Azure subscription 1` / compte gratuit) qui ne supporte pas les managed online endpoints en `francecentral`.

Le **Batch Endpoint** a été choisi comme alternative architecturale justifiée : il est supporté sur toutes les souscriptions, adapté aux prédictions sur de grands volumes de données, et cohérent avec le cas d'usage réel (scoring offline de campagnes de recommandation).

### Bug diagnostiqué en scoring batch

Le test du batch endpoint (`test_batch_endpoint.py`) échoue sur la colonne `seq_embs` dans `run()` de `batch_score.py`. La cause racine est identifiée précisément :

- En entraînement (`train.py`), le padding des séquences d'historique utilisateur est géré par le `DataLoader` (collate implicite sur des tenseurs de taille fixe `[batch, 50, 128]`).
- En scoring batch, `run(mini_batch)` reçoit un DataFrame où chaque ligne contient une séquence de longueur variable (l'historique réel de l'utilisateur : 3 items pour l'un, 7 pour l'autre). L'empilement vectorisé `np.stack(...)` échoue car les tailles diffèrent.
- **Piste de résolution** : appliquer le même padding à 50 que dans `CTRDataset.__getitem__` avant l'empilement, ou traiter chaque ligne individuellement en boucle (comme `score.py` le fait déjà pour le scoring online).

### AUC 0.7752 vs 0.51 — Explication

- **0.7752** : obtenu sur Kaggle (GPU Tesla T4, 15 epochs, données complètes bien structurées).
- **~0.51** : obtenu sur Azure ML (CPU Standard_DS3_v2, run limité à 5 epochs). Deux facteurs expliquent la différence :
  1. **Absence de GPU** : l'entraînement sur CPU est ~10x plus lent, forçant un arrêt prématuré.
  2. **Bug item_seq** (corrigé en cours de route) : `item_seq.parquet` contient 6M lignes (pas 1M users), avec plusieurs lignes par utilisateur. Le chargement naïf écrasait la séquence à chaque ligne. Corrigé en chargeant par batches de 500K lignes avec `pyarrow.ParquetFile.iter_batches()`.

### Fichiers MLOps

```
.github/workflows/deploy.yml   # CI/CD : retrain ou deploy-only
src/train.py                   # Entraînement + MLflow tracking
src/score.py                   # Scoring online (init + run)
src/batch_score.py             # Scoring batch (init + run(mini_batch))
pipelines/training_pipeline.py # Soumission job + enregistrement modèle
pipelines/deploy_batch.py      # Création batch endpoint + deployment
generate_batch_sample.py       # Génère data/batch_test_sample.parquet
test_batch_endpoint.py         # Soumet un job batch et affiche les résultats
monitor.py                     # Détection de dérive PSI + KS
environments/conda.yml         # Environnement conda reproductible
```

---

## 👤 Author

**Oumniya Moutaouakil**
- Master's Student in Advanced Machine Learning & Multimedia Intelligence.
- GitHub: [@oumniya03](https://github.com/oumniya03)
- Project: [Projet-competition](https://github.com/oumniya03/Projet_competition.git)





