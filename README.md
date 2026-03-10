# Détection de régimes latents pour les liquidity providers en DeFi - Cadre mathématique

### Une approche probabiliste du market-making dynamique sur Uniswap V3

---

## 1. Introduction : formalisation du problème

Considérons un market-maker automatisé (AMM) tel qu'Uniswap V3, où un fournisseur de liquidité (FL) peut concentrer son capital dans une plage de prix personnalisée. La rentabilité et le risque de cette stratégie dépendent de manière critique du *comportement* du marché — le prix évolue-t-il lentement, permettant l'accumulation de frais, ou est-il volatile, exposant le FL à des pertes impermanentes (PI) ?

Ce comportement de marché n'est **pas directement observable**. Je postule qu'il est gouverné par un ensemble fini de **régimes latents (cachés)**, notés $Z_t \in \{1,2,\dots,K\}$. Dans mon application, j'interprète $K=3$ régimes comme suit :
- $Z_t = 1$ **(Goldilocks)** : faible volatilité, capture élevée de frais — idéal pour la liquidité concentrée.
- $Z_t = 2$ **(Tendanciel)** : mouvements directionnels avec volatilité modérée.
- $Z_t = 3$ **(Toxique)** : forte volatilité, risque de sélection adverse — dangereux pour les FL passifs.

J'observe cependant un vecteur de variables de marché à chaque pas horaire $t$ :

$$\mathbf{X}_t = \big[ \text{log_rendement}_t,\; \text{nombre_trades}_t,\; \text{frais_totaux_usd}_t,\; \text{vol_réalisée}_t \big]^T.$$

Mon objectif est d'inférer la séquence de régimes cachés la plus probable $Z_{1:T}$ étant donné la séquence observée $\mathbf{X}_{1:T}$.

> 📓 La construction et le nettoyage de ce vecteur de features sont détaillés dans [`notebooks/01_data_prep.ipynb`](notebooks/01_data_prep.ipynb).

---

## 2. Le modèle de Markov caché : un modèle de probabilité jointe

Un MMC est défini par deux processus stochastiques :

### 2.1 La chaîne de Markov latente (dynamique des régimes)
La séquence d'états cachés $\{Z_t\}$ suit une chaîne de Markov du premier ordre avec une matrice de transition homogène dans le temps $\mathbf{A} = (A_{ij})_{K\times K}$ :

$$P(Z_t = j \mid Z_{t-1} = i, Z_{t-2}, \dots) = P(Z_t = j \mid Z_{t-1} = i) = A_{ij},$$

avec $\sum_{j=1}^K A_{ij}=1$ pour chaque ligne $i$. La distribution initiale des états est $\boldsymbol{\pi} = (\pi_1,\dots,\pi_K)$ où $\pi_i = P(Z_1 = i)$.

### 2.2 Le processus d'émission (observations)
Étant donné l'état caché courant $Z_t$, l'observation $\mathbf{X}_t$ est indépendante de toutes les observations et états passés et futurs :

$$P(\mathbf{X}_t \mid Z_t = k, \mathbf{X}_{1:t-1}, Z_{1:t-1}) = P(\mathbf{X}_t \mid Z_t = k).$$

### 2.3 La distribution jointe
Ces deux hypothèses permettent de factoriser la probabilité jointe complète :

$$P(Z_{1:T}, \mathbf{X}_{1:T}) = P(Z_1) \prod_{t=2}^{T} P(Z_t \mid Z_{t-1}) \prod_{t=1}^{T} P(\mathbf{X}_t \mid Z_t).$$

> 📓 L'entraînement du MMC via Baum–Welch, la sélection de $K$ par BIC et le décodage de Viterbi sont implémentés dans [`notebooks/03_hmm.ipynb`](notebooks/03_hmm.ipynb).

---

## 3. L'hypothèse d'émission gaussienne : un choix de modélisation

Pour rendre le modèle tractable, je suppose que la distribution d'émission pour chaque état est une loi normale multivariée à covariance diagonale :

$$\mathbf{X}_t \mid Z_t = k \;\sim\; \mathcal{N}(\boldsymbol{\mu}_k,\; \boldsymbol{\Sigma}_k), \qquad \boldsymbol{\Sigma}_k = \mathrm{diag}(\sigma_{k,1}^2,\dots,\sigma_{k,d}^2).$$

### 3.1 Estimation des paramètres (étape M)
Sous cette hypothèse gaussienne, les estimateurs du maximum de vraisemblance des paramètres, étant données les probabilités d'état postérieures $\gamma_t(k) = P(Z_t = k \mid \mathbf{X}_{1:T})$, sont :

$$\hat{\boldsymbol{\mu}}_k = \frac{\sum_{t=1}^T \gamma_t(k)\,\mathbf{x}_t}{\sum_{t=1}^T \gamma_t(k)}, \qquad \hat{\sigma}_{k,j}^2 = \frac{\sum_{t=1}^T \gamma_t(k)\,(x_{t,j} - \hat{\mu}_{k,j})^2}{\sum_{t=1}^T \gamma_t(k)}.$$

### 3.2 Sélection du modèle (BIC)
Pour choisir le nombre d'états cachés $K$, j'utilise le Critère d'Information Bayésien :

$$\text{BIC} = -2\ln\hat{L} + m\ln T,$$

où $\hat{L}$ est la vraisemblance maximisée et $m = K(K-1) + (K-1) + 2Kd$ est le nombre de paramètres libres.

> 📓 La grille de recherche sur $K$ et les courbes BIC sont tracées dans [`notebooks/03_hmm.ipynb`](notebooks/03_hmm.ipynb).

---

## 4. Vérification des hypothèses fondamentales

Avant de faire confiance aux inférences du modèle, il convient de vérifier ses hypothèses sous-jacentes :

| **Hypothèse** | **Implication mathématique** | **Méthode** |
| :--- | :--- | :--- |
| **Stationnarité** | Matrice de transition $\mathbf{A}$ constante dans le temps | Tests ADF et KPSS |
| **Émissions Gaussiennes** | Vraisemblance et BIC bien spécifiés | Test de Jarque–Bera, graphiques Q-Q |
| **Indépendance Conditionnelle** | Covariance diagonale $\boldsymbol{\Sigma}_k = \mathrm{diag}(\dots)$ | Test de sphéricité de Bartlett |
| **Propriété de Markov du 1er ordre** | $Z_t \perp Z_{t-2} \mid Z_{t-1}$ | Test du rapport de vraisemblance de Billingsley |

> 📓 L'ensemble de ces tests statistiques est détaillé et exécuté dans [`notebooks/00_stats.ipynb`](notebooks/00_stats.ipynb).

---

## 5. Du modèle à la stratégie : interprétation des résultats

Après estimation des paramètres via Baum–Welch, la séquence d'états la plus probable est retrouvée par l'**algorithme de Viterbi** :

$$\hat{Z}_{1:T} = \arg\max_{Z_{1:T}} P(Z_{1:T} \mid \mathbf{X}_{1:T}; \hat{\boldsymbol{\theta}}),$$

où $\hat{\boldsymbol{\theta}} = \{\boldsymbol{\pi}, \mathbf{A}, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}$.

Chaque état inféré correspond à une condition de marché distincte :
- **Goldilocks** (faible `vol_réalisée`, élevé `frais_totaux_usd`) : **réduire la plage** pour maximiser la capture de frais.
- **Tendanciel** (volatilité modérée, `log_rendement` directionnel) : plage équilibrée.
- **Toxique** (forte `vol_réalisée`, faible efficacité des frais) : **élargir la plage** ou se retirer temporairement pour atténuer les pertes impermanentes et le LVR.

> 📓 Le labellisation heuristique des régimes et l'estimation de la matrice de transition sont explorés dans [`notebooks/02_heuristic_markov.ipynb`](notebooks/02_heuristic_markov.ipynb).

---

## 6. Conclusion

J'ai formalisé le problème de détection des régimes de marché latents pour un pool Uniswap V3 à l'aide d'un Modèle de Markov Caché. En spécifiant les variables aléatoires, la structure du modèle et les hypothèses sous-jacentes, j'ai construit un cadre mathématique rigoureux qui connecte la théorie classique des probabilités à une application DeFi moderne. Les hypothèses ont été examinées de manière critique à travers des tests statistiques, et les états décodés finaux fournissent des recommandations actionnables pour les fournisseurs de liquidité.

Ce travail démontre comment les modèles probabilistes de séries temporelles peuvent être mis à profit pour améliorer la prise de décision en finance décentralisée.

---

## Structure du dépôt

```
notebooks/
├── 01_data_prep.ipynb          # Chargement des données brutes, agrégation en features horaires
├── 00_stats.ipynb              # Tests statistiques des hypothèses du MMC
├── 02_heuristic_markov.ipynb   # Labellisation heuristique et matrice de transition
└── 03_hmm.ipynb                # Entraînement du MMC, sélection de K, décodage de Viterbi
```

---

## Installation & reproductibilité

> **Note :** Les fichiers de données (`data/`, `usdc_weth_pool.parquet/`) ne sont pas versionnés. Ils doivent être regénérés localement en suivant les étapes ci-dessous.

```bash
# 1. Cloner le dépôt
git clone https://github.com/<your-username>/hmm-modelling-for-lps.git
cd hmm-modelling-for-lps

# 2. Créer et activer un environnement virtuel
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

**4. Configurer les secrets** — créer un fichier `.env` à la racine du projet :

```
HYPERSYNC_BEARER_TOKEN=your_token_here
```

**5. Regénérer les données** — exécuter dans l'ordre :

| Étape | Script / Notebook | Sortie |
|---|---|---|
| Extraction on-chain | `python extract.py` | `usdc_weth_pool.parquet/` |
| Préparation des features | `notebooks/01_data_prep.ipynb` | `data/hourly_features.parquet` |
| Tests statistiques | `notebooks/00_stats.ipynb` | — |
| Labellisation heuristique | `notebooks/02_heuristic_markov.ipynb` | `data/hourly_features_labelled.parquet` |
| Modèle HMM | `notebooks/03_hmm.ipynb` | — |
