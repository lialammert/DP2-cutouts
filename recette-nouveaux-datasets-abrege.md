---
title: Recette abrégée - jeu de données réel, galaxies isolées
---

# Recette abrégée

Version courte de `recette-nouveaux-datasets.md`, qui porte les justifications. Ici,
une phrase par geste et une phrase par limite.

## Avant de commencer

| besoin | pourquoi |
|---|---|
| un compte sur un centre de calcul | découper et mesurer les vignettes |
| un compte sur la plateforme de Rubin, avec les chaînes de traitement et le butler | l'étape de préparation de `copaings` en a besoin |
| une seconde machine à internet ouvert | le catalogue d'Euclid est souvent bloqué par les pare-feu de centre de calcul |
| un jeton d'accès à la plateforme de Rubin, dans `~/.rsp_token`, droits 600 | télécharger les images ; ne jamais l'afficher ni le versionner |
| `copaings` et `friendly` | l'appariement, les recouvrements d'ellipses et la blending entropy |
| `lsdb`, `pyvo`, `galsim`, `numpy`, `scipy`, `astropy` | lire les catalogues et mesurer les formes |

**Les deux sources.** Rubin fournit les pixels : six images profondes par zone de
ciel, une par couleur, 0,2 seconde d'arc par pixel, vignettes de 59 × 59 pixels.
Euclid fournit le catalogue : il voit deux fois plus fin, dit si une source est une
galaxie, et donne le redshift.

**La règle.** C'est Euclid qui certifie l'isolement. Rubin peut seulement opposer un
veto — on ne lui demande jamais de garantir qu'il n'y a pas de voisine, puisqu'il
répondrait « rien » là où il ne la distingue pas.

**Étape 1.** On télécharge le catalogue d'Euclid zone par zone, de façon cumulative,
jusqu'à atteindre le nombre de galaxies voulu, sans jamais retirer une zone déjà
tirée. L'objectif est un nombre de galaxies en fin de chaîne, pas un nombre de zones.

## Les huit coupures, dans l'ordre

Les deux premières ne lisent que les catalogues : elles s'appliquent **avant tout
téléchargement d'image**. Les suivantes portent sur les pixels.

| # | ce qu'on fait | limite admise |
|---|---|---|
| 1 | Garder les sources qu'Euclid ne classe ni ponctuelles ni artefacts. | Près de la limite de détection, une galaxie compacte faible est classée ponctuelle et perdue ; aucune vérité de référence ne permet de le corriger. |
| 2 | Faire tourner `copaings` sur le catalogue de Rubin contre celui d'Euclid — regroupement amis des amis avec une longueur de lien de **3 secondes d'arc**, puis test de recouvrement d'ellipses — et ne garder que les objets de blending entropy `S_b = 0`, c'est-à-dire appariés un à un après ce test. | Cette seule condition remplace l'appariement entre les deux relevés, le critère d'isolement et le rejet des morceaux de galaxies découpées. Elle retire de l'ordre de 30 % des galaxies, réparable par repondération en magnitude × log taille **à condition de garder le catalogue d'avant la coupure**. Mais « isolée » veut dire isolée au sens des deux relevés : un mélange qu'aucun des deux ne résout survit, et c'est le plancher de 0,01 à 0,02 en ellipticité. Et une galaxie grumeleuse que l'un des relevés a découpée en morceaux est rejetée à tort, biais en morphologie que la repondération ne répare pas. |
| 3 | Découper la vignette dans la zone que le catalogue attribue à la galaxie, et exiger 29 pixels de marge des quatre côtés, en comparaison d'entiers. | On perd le bord extérieur du relevé, perte non corrélée aux propriétés des galaxies ; **à quantifier**. |
| 4 | Rejeter si un seul pixel de la boîte est marqué non exposé, dans n'importe laquelle des six couleurs. | Sévérité choisie : un pixel non exposé loin de la galaxie ne gênerait probablement pas ; **à quantifier**. |
| 5 | Rejeter si un pixel abîmé — saturation, rayon cosmique, bord de capteur — tombe dans **`k` fois l'ellipse de la coupure 2**, la même que celle du test de recouvrement, avec `k` de l'ordre de 3. | `k` doit être calibré là où le déplacement d'ellipticité causé par un pixel faux retombe sous 0,02 ; la liste des marqueurs est un choix ; **à quantifier**. |
| 6 | Garder la galaxie si `i_cModelFlux / i_cModelFluxErr` dépasse 5, et enregistrer cette valeur. | Plancher non testé ; et la mesure de forme ne sélectionne jamais, donc la performance en forme ne sera chiffrée que sur les galaxies mesurables, d'où la nécessité d'une seconde métrique en pixels ; **à quantifier**. |
| 7 | Rejeter si les six solutions astrométriques n'envoient pas la position de la galaxie sur le même pixel. | Ne doit jamais déclencher : si cela arrive souvent, les fichiers sont mal appariés ; **à quantifier** comme diagnostic. |
| 8 | Découper une vignette de diagnostic deux fois plus grande, mesurer la galaxie dans les ouvertures **circulaires** de 29 et 58 pixels de rayon, rejeter si l'ellipticité diffère de plus de 0,02. | Jamais appliquée : c'est la seule coupure dont on ne connaît pas l'ordre de grandeur de l'effet, **à évaluer en premier**. |

**Pourquoi 3 secondes d'arc et pas 2.** C'est la valeur recommandée par `copaings`,
mesurée et non choisie : à 2 secondes d'arc, environ 10 % des mélanges qu'Euclid
identifie passent au travers. Attention toutefois : la longueur de lien ne fait que
rassembler les candidats, et c'est le recouvrement des ellipses qui décide. Le rayon
d'isolement effectif est donc celui des ellipses, bien plus petit que 3 secondes d'arc.

**Une seule ellipse pour toute la chaîne.** `copaings` définit l'ellipse par les
moments d'ordre deux du flux en bande *i*, avec le facteur $2\ln 2$ qui la place à la
demi-lumière, et les moments des détections portent déjà la tache instrumentale — donc
l'ellipse d'une galaxie minuscule ne descend jamais sous la taille de la tache. C'est
cette même ellipse qui sert à la coupure 5 et au masquage, au lieu d'un rayon en
secondes d'arc : son aire croît avec la taille apparente, ce qui introduit une
dépendance en taille, réparable par la repondération ; et à rayon effectif fixé son
aire ne dépend pas de l'ellipticité, donc elle n'introduit **pas** le seul biais qu'on
ne sait pas réparer.

**Deux gestes techniques à ne pas rater.** La solution astrométrique rend (colonne,
ligne) alors qu'un tableau s'indexe [ligne, colonne] : les inverser transpose la
vignette et échange les deux composantes de l'ellipticité. Et les conventions
d'origine diffèrent d'un pixel entre FITS et les tableaux, ce qui décale le centrage.

## Le masquage des compagnes, et la décision qui va avec

Toute compagne présente dans la boîte sans appartenir au groupe de la cible est
masquée sur l'étendue de son ellipse, et la zone masquée est remplie par un morceau de
ciel vide découpé dans la même zone et la même couleur — jamais du bruit blanc. L'image
montre alors une galaxie seule, ce qu'elle est censée être. Même traitement dans la
vignette de diagnostic de la coupure 8, de sorte que la seule différence entre les deux
ouvertures soit leur taille.

**Et une décision, qui en fait aussi une coupure.** On mesure l'ellipticité de la cible
avant et après masquage, et on enregistre l'écart. **Si l'écart dépasse 0,02, on écarte
la vignette** ; sinon on la garde masquée. Le masquage ne rachète pas une
contamination : si une compagne déplace la forme, c'est que sa lumière entre dans la
zone où la cible est mesurée, et masquer cette zone efface la cible avec elle.

La coupure 2 ne dispense pas de ce test. Son rayon effectif est celui du recouvrement
des ellipses, à quelques dixièmes de seconde d'arc, alors que la pondération de la
mesure de forme s'étend deux à trois fois plus loin : il subsiste donc une bande où une
compagne ne touche pas la cible mais déplace sa forme. C'est cette bande que le test
attrape.

**Et la fraction de pixels masqués est enregistrée par vignette**, pour vérifier après
coup que la performance du réseau n'en dépend pas.

## À enregistrer sans couper

| grandeur | pourquoi |
|---|---|
| blending entropy `S_b` de chaque objet | garder la valeur, et pas seulement la condition `S_b = 0`, permet de durcir ou d'assouplir l'isolement sans reconstruire le jeu |
| drapeau de qualité du redshift d'Euclid | une galaxie à bonne image et redshift douteux reste un bon exemple pour la forme ; le drapeau se lit tâche par tâche à l'entraînement |
| taille et deux composantes d'ellipticité de la tache instrumentale, par couleur | la tache de Rubin n'est pas ronde, celle des simulations l'est, et la fuite correspondante a déjà été mesurée ; sans ces colonnes elle n'est ni démontrable ni corrigible |
| facteur de résolution galaxie/tache | dit quand l'ellipticité mesurée est dominée par l'instrument ; sert de poids, jamais de critère |
| rapport signal sur bruit, écart d'ellipticité au masquage, écart entre les deux ouvertures de la coupure 8, demi-axes et angle de l'ellipse, profondeur locale de l'empilement | tous les seuils restent ainsi révisables sans reconstruire le jeu |
