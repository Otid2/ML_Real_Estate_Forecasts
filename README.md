# ML_Real_Estate_Forecasts

Analyse comparative d'une méthode des moindres carrés ordinaires (MCO) et de méthodes de régularisation (Ridge, LASSO) pour la prédiction immobilière.

**Auteurs :** Ryan Zahni & Anaïs Kagan

----

## Problématique

Dans quelle mesure l'ajout d'une pénalité de régularisation (Ridge, LASSO) permet-elle d'améliorer la méthode des moindres carrés dans la prédiction des prix de l'immobilier, tout en facilitant l'identification des facteurs qui influencent le plus le marché ?

----

## Le Jeu de Données

On utilise le dataset [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) de Kaggle.

Le problème principal ici, c'est la forte multicolinéarité entre les variables explicatives (par exemple, la surface habitable et le nombre de pièces). Ça rend le MCO instable et ça pousse direct à l'overfitting.

----

## Méthode

### 1. Préparation et Nettoyage
* Suppression des colonnes poubelles / quasi-constantes (ex: `Alley`, `PoolQC`).
* Renommage des colonnes en français et remplacement des NaNs par la médiane.
* Encodage : One-Hot Encoding (`drop_first=True`) pour les variables nominales à basse cardinalité, Target Encoding pour le quartier, et mapping ordinal pour les notes de qualité.

### 2. Implémentations (du manuel au scikit-learn)
* **MCO :** Calcul matriciel direct avec numpy ($\hat{\theta} = (X^T X)^{-1}X^T y$). On montre que si on réduit la taille du train set ($n=500$), la matrice est mal conditionnée, le déterminant s'approche de 0, les coefficients explosent et le $R^2$ devient négatif.
* **MCO + ACP :** On compresse l'info sur 95% des axes orthogonaux pour virer le bruit et casser la colinéarité. Ça stabilise le modèle, mais on perd l'interprétabilité métier des variables de départ.
* **Ridge ($L^2$) :** On ajoute $\lambda I$ sur la diagonale de $X^T X$, ce qui rend la matrice inversible et bloque l'explosion des poids. Testé en manuel (via cross-validation) et avec `RidgeCV`.
* **LASSO ($L^1$) :** Ajout d'une pénalité sur la valeur absolue des coefficients (via `LassoCV`). Ça permet en plus de mettre certains coefficients à zéro pour faire de la sélection de features.

----

## Résultats

* **Robustesse :** Ridge et LASSO gardent un bon score ($R^2 \approx 0.85$ en complet, $\approx 0.80$ en réduit à $n=500$) là où le MCO plante complètement.
* **Sélection (LASSO) :** Le modèle a réduit à 28 variables pertinentes. Il a par exemple éliminé l'année de construction au profit de la qualité globale pour éviter les doublons d'information.
* **Interprétation :** Les coefficients montrent des trucs logiques (surface, qualité = positif), mais aussi des effets de structure intéressants (le nombre de chambres a un impact négatif à surface constante, ce qui traduit simplement des pièces plus petites).

----

## Stack Technique
* Python
* `pandas`, `numpy`, `category_encoders`
* `scikit-learn` (`StandardScaler`, `PCA`, `KFold`, `RidgeCV`, `LassoCV`)
* `matplotlib`, `seaborn`
