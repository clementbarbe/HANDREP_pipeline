# Analyse scientifique — version 2

## Exécution et conservation des données

```bash
conda activate MOTORPLANNING
python -m pipeline.tools.predict --output-dir data/nn_outputs_scientific
python run.py --prediction-dir data/nn_outputs_scientific --output-dir data/processed_scientific --bootstrap-samples 2000 --seed 42
python -m unittest discover -s tests -v
```

Cette commande conserve les anciennes sorties dans `data/processed` et écrit
les nouvelles dans `data/processed_scientific`. Les images, les annotations et
les prédictions d'entrée ne sont pas modifiées. Le recalcul est systématique :
la simple présence d'un ancien CSV ne suffit plus à sauter une analyse.
L'inférence en lot écrit les nouvelles prédictions dans `data/nn_outputs_scientific`,
avec l'empreinte des poids, des images et du CSV, les dimensions du réseau et
la configuration d'exécution. `--prediction-dir` permet de choisir explicitement
ces nouvelles prédictions. Les fichiers d'origine dans `data/nn_outputs` restent
disponibles. Si un manifeste de génération existe, une divergence de son empreinte
avec le CSV de prédictions est refusée.

Chaque session possède un manifeste JSON : paramètres, versions, empreintes
SHA-256 des entrées et du code, nombre d'essais valides, calibration et réserves
scientifiques. Un manifeste `complete` n'est écrit qu'après les exports et les
figures. `run_report.json` recense les sessions réussies et les échecs. Un échec
produit un code de sortie non nul.

## Comparaison des conditions

Dans les deux conditions, la longueur estimée est :

`norme(moyenne(position Z2) - moyenne(position Z1))`.

Les distances de chaque paire restent exportées comme mesures complémentaires ;
leur moyenne n'est plus utilisée comme estimateur principal dans une seule
condition. Les paires incomplètes ne contribuent pas à la longueur couplée ;
leur endpoint valide peut contribuer à la carte globale. Les effectifs de chaque
zone et de chaque condition sont exportés.

Les IC à 95 % sont des quantiles de 2 000 réplications bootstrap par défaut.
Les deux endpoints d'une paire sont rééchantillonnés ensemble. Les essais non
couplés sont rééchantillonnés séparément dans chaque zone. Le contraste est
calculé entre les longueurs des cartes moyennes rééchantillonnées. Les IC des
longueurs, largeurs et indices de forme de la carte globale conservent aussi les
paires complètes. Aucun IC n'est produit avec moins de trois répétitions par
endpoint ou trois paires.

**Ces IC sont exploratoires** : ils ne représentent que l'incertitude
d'échantillonnage intra-session, conditionnellement à la référence et à la
calibration observées. Ils n'intègrent ni l'incertitude d'annotation, ni celle du
réseau, ni celle de la calibration, ni une dépendance temporelle arbitraire.
Une comparaison temporelle ou causale exige un protocole adapté.

Pour utiliser les conditions réellement prévues, fournir :

```csv
trial,condition,pair_id
1,coupled,p001
2,coupled,p001
3,uncoupled,
```

Chemin : `data/design/S1_ses-1_design.csv`. Les mêmes colonnes peuvent être
directement présentes dans le CSV des prédictions. Une seule source est acceptée.
Une paire complète doit contenir deux essais consécutifs du même doigt avec
une zone Z1 et une zone Z2. Une paire dont un essai manque reste identifiée comme
prévue couplée, sans être assimilée à un essai non couplé.

Sans métadonnées, les successions sont détectées de façon non chevauchante,
avec une différence de numéro d'essai exactement égale à 1. Un essai manquant,
un changement de session, sujet ou bloc empêche un appariement artificiel.
`coupling_source=inferred_adjacency_exploratory` précise qu'il s'agit d'une
association observée, pas d'une manipulation expérimentale démontrée.

## Coordonnées et calibration

Les coins du damier sont associés à une grille physique dont le côté des carrés
est connu. Le plan de la main est rectifié pour rendre les distances métriques,
y compris lorsque les échelles horizontale et verticale de l'image diffèrent.
Les réponses et les repères anatomiques sont exprimés dans le même plan.

La taille physique réelle des carrés doit être mesurée ; la valeur par défaut
reste celle du projet, 3 cm. Elle peut être changée avec `--square-size-cm`.
Les dimensions du damier sont configurées dans `pipeline/config/settings.py`.
La caméra doit rester fixe et les coins des deux images doivent représenter les
mêmes positions physiques dans le plan horizontal. Un déplacement latéral ou
une rotation du damier entre les hauteurs compromet la correction de localisation.
La symétrie du damier ne permet pas de vérifier automatiquement cette condition.

Le seuil technique par défaut est de 2 px pour l'erreur moyenne d'ajustement et
de contrôle sur des coins laissés hors ajustement ; au moins 90 % des coins
doivent être retenus par RANSAC. Le contrôle métrique utilise également des coins
laissés hors ajustement. Ces contrôles utilisent les mêmes photos de calibration :
ils ne remplacent pas une validation physique indépendante couvrant la zone de
pointage. Les seuils sont des paramètres techniques configurables et non une
garantie de validité du dispositif.

Une calibration optique indépendante peut être fournie dans
`data/calibration/camera.npz`, avec les tableaux `camera_matrix` et `dist_coeffs`.
Sinon, l'absence de correction de distorsion est déclarée dans le manifeste.
Le modèle suppose des plans et ne reconstruit pas la géométrie 3D des doigts.

Une calibration propre à une session peut être placée dans
`data/calibration/sub-S1/ses-1/` (`table.png`, `plateau.png`, éventuellement
`camera.npz`). Elle est prioritaire ; un dossier de session incomplet n'est pas
remplacé silencieusement par la calibration générale. Sinon la calibration
générale est partagée et ce choix est déclaré dans les réserves du manifeste.
La résolution des images brutes disponibles est vérifiée par rapport à celle
de la calibration ; une incompatibilité provoque un arrêt. Les empreintes des
images disponibles sont également conservées dans le manifeste.

La calibration invalide provoque un arrêt. `--allow-fallback` autorise
explicitement une approximation par changement d'échelle autour du centre de
l'image. Cette approximation ne produit aucune valeur en centimètres et ne doit
pas être mélangée aux résultats calibrés pour une analyse confirmatoire.

Colonnes principales :

| Colonnes | Signification |
|---|---|
| `x_est,y_est` | Prédiction originale, pixels de l'image du plateau |
| `x_real_image,y_real_image` | Annotation originale, pixels de l'image de la main |
| `x_est_table_px,y_est_table_px` | Réponse projetée dans l'image du plan de la main |
| `x_est_corr,y_est_corr,x_real,y_real` | Coordonnées utilisées dans les analyses, pixels du plan rectifié |
| `coordinate_space,correction_method` | Repère et méthode effectivement utilisés |
| `error_*_cm` | Erreurs physiques après rectification, uniquement si calibration acceptée |
| `error_*_norm` | Erreurs divisées par le rayon RMS des dix repères anatomiques |

La normalisation RMS est indépendante de la rotation de la main et du nombre
de répétitions de chaque repère ; elle remplace la diagonale de boîte englobante.
Les valeurs normalisées de version 1 et de version 2 ne sont pas interchangeables.

## Références et qualité des observations

Les doublons d'essai, les identifiants manquants, les zones/doigts inconnus et les
références anatomiques incomplètes sont refusés. Les prédictions non finies ou
hors image restent dans le CSV avec `qc_valid=False` et un motif d'exclusion.
Elles sont exclues des calculs numériques sans rapprocher artificiellement les
essais autour d'elles. Les réponses comportementales éloignées mais dans l'image
ne sont pas automatiquement supprimées comme « aberrantes ».

Annoter les deux photos de référence avec `python -m pipeline.tools.annotate`
permet de mesurer le déplacement réel des repères. Si plusieurs photos sont
annotées, sélectionner explicitement la référence de calcul :

```bash
python run.py --reference-image S1_REF1.jpg --output-dir data/processed_scientific
```

Les références ne sont jamais automatiquement moyennées ni interpolées dans le
temps. L'audit exprime les déplacements par rapport à la photo sélectionnée,
sans supposer que les noms REF1/REF2 prouvent leur rôle temporel. Un critère de
déplacement peut être fixé avec `--max-reference-motion-cm` après justification
par le protocole et la précision du dispositif. Sans annotation de la seconde
photo, la stabilité physique de la main reste non évaluée.

Pour évaluer le réseau, fournir un échantillon indépendant annoté manuellement :

```csv
filename,x_manual,y_manual
S1_T1_index_Z1.jpg,923.5,412.0
```

Chemin : `data/validation/S1_ses-1_pointer.csv`. L'export donne les biais X/Y,
l'erreur médiane et le 95e percentile par repère, dans les pixels originaux.
Ce contrôle ne prouve pas automatiquement l'indépendance du jeu d'entraînement.
Une double annotation et une évaluation par sujet/session indépendante restent
nécessaires pour établir l'incertitude de la mesure.

Le décalage de RefineNet est désormais appliqué au centre réel du recadrage,
ce qui évite un décalage artificiel lorsque le crop est déplacé près d'un bord.
Les anciens CSV de prédiction ne sont pas remplacés automatiquement : il faut
relancer l'inférence pour bénéficier de cette correction. La convention des
offsets doit rester cohérente avec celle utilisée à l'entraînement.

## Analyses ajoutées et interprétation

Le repère anatomique utilise le majeur pour l'axe proximodistal et la direction
auriculaire→index, orthogonalisée, pour l'axe radioulnaire. Les signes positifs
indiquent le bout des doigts et le pouce. Le biais est le vecteur moyen d'erreur ;
la variabilité est la distance RMS des réponses à leur moyenne. On vérifie
`erreur_RMS² = norme(biais)² + erreur_variable²`. Les ellipses représentent
la dispersion des réponses à 1 écart-type, pas un IC de leur moyenne.

La décomposition géométrique sépare translation, rotation rigide, taille
uniforme et étirements affines. Ces ajustements ont lieu sur les repères moyens
avec le même poids par repère. Les cartes TPS utilisent une rotation correcte,
sans supprimer les différences de taille. Leur régularisation est indépendante
de l'unité des coordonnées. Ce sont des interpolations exploratoires entre dix
repères, sans démonstration d'une carte neurale.

Les régressions temporelles comprennent des intercepts par repère, le numéro
d'essai centré et, lorsqu'il varie, le statut de couplage. Une seconde régression
ajoute la position de la cible précédente par rapport à la cible actuelle.
Les essais non consécutifs ne contribuent pas à cette association séquentielle.
Les coefficients sont descriptifs, sans tests de significativité sur des essais
supposés indépendants et sans conclusion causale.
Les pentes sont exprimées en cm/essai lorsque la calibration est disponible ;
le coefficient associé à la cible précédente est sans dimension. Les unités
figurent dans chaque ligne exportée.

La synthèse de cohorte moyenne d'abord les sessions de chaque participant,
puis donne le même poids à chaque participant. Les méthodes de correction et
les définitions de couplage différentes ne sont pas regroupées. Un bootstrap
de participants est fourni à partir de cinq sujets, avec statut exploratoire ;
ce seuil n'est pas un calcul de puissance. Avec les deux jeux locaux actuels,
la synthèse reste descriptive. La comparabilité de la main testée, des sessions
et des conditions doit être vérifiée dans les métadonnées d'acquisition.

## Sorties supplémentaires par sujet/session

- `results_coupling_comparison.csv` : contraste couplé moins non couplé et IC.
- `results_finger_lengths_by_direction.csv` : Z1→Z2 et Z2→Z1 séparés.
- `results_bias_precision.csv` : biais anatomiques et dispersion.
- `results_geometry_decomposition.csv` : translation, rotation, taille, résidus.
- `results_temporal_models.csv` : dérive et association avec la cible précédente.
- `results_reference_audit.csv` : photos annotées, manquantes et déplacements.
- `results_pointer_validation_*.csv` : validation du réseau si fournie.
- `results_shape_indices.csv` : indice de forme avec intervalle conditionnel.
- `quality_report.txt` et `manifest.json` : qualité et traçabilité.
- Figure 09 : estimateur identique et IC dans les deux conditions, contraste.
- Figure 10 : biais et variabilité dans le repère anatomique.
- Figure 11 : erreurs anatomiques signées au cours de l'acquisition.

## Références méthodologiques

- [Longo et Haggard, 2010 — reconstruction des cartes de la main](https://pmc.ncbi.nlm.nih.gov/articles/PMC2900654/).
- [Ingram et al., 2019 — biais directionnels et dérive temporelle](https://pubmed.ncbi.nlm.nih.gov/30653568/).
- [Peviani et Bottini, 2020 — interprétation des erreurs de localisation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0236416).
- [OpenCV — homographies et correction de perspective](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html).
