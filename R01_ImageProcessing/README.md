# Traitement d'images

## 1. Concept de base
### 1.1 Définition d'une image
* L'image est un objet qui peut directement ou indirectement agir sur les yeux et produire la perception visuelle.  
* L'image est une distribution des réflexions optiques, et une impression dans le cerveau générée par le système d'organe visuel.  
* L'image est une réflexion objective de la nature.  
> Exemples: ce que l'on voit, photo, télé, cinéma
### 1.2 Classification des images
1. Par le sens visuel
   * Image visible
   * Image invisible: image infrarouge, image RSO (Radar à synthèse d'ouverture / Synthetic-aperture radar)
2. Par la continuité sur les coordonnées spatiales et la luminosité
   * Image analogue
   * __Image numérique__

## 2. Image numérique
### 2.1 Obtension d'une image numérique
* Image continue, image discrète  
* Numériser une image est de discrétiser une image continue.  
* La numérisation d'image contient 2 parties :
    * Echantillonnage : discrétisation en espace  
    * Quantification : discrétisation en luminosité  
* Une image numérique peut être décrite par une matrice ou un tableau.
    * Résolution : plus haute la résolution, plus les pixels qui composent une image, plus grand le fichier de l'image  
	* Profondeur : plus profond le pixel, plus les chiffres qui expriment la couleur et la luminosité de l'image, plus grand le fichier de l'image  

### 2.2 Voisinage d'une image numérique
* ___N4(p)___: (x+1, y), (x-1, y), (x, y+1), (x, y-1)  
* ___ND(p)___: (x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)  
* ___N8(p)___: ___N4(p) + ND(p)___  

### 2.3 Classification d'images
* Image noir et blanc
    * Pixel en valeur binaire : 0, 1  
* Image en niveaux de gris
    * Il n'y a que la luminosité mais pas de couleur.  
* Image couleur
    * Il n'y a pas seulement la luminosité mais aussi la couleur.  

## 3. Segmentation d'image
### 3.1 Valeur de seuil
> retval, dst = cv.threshold(src, thresh, maxval, type[, dst])

___Méthode d'OSTU___
> dst = cv.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst])

Le seuil qui minimise la variance intra-classe est recherché à partir de tous les seuillages possibles.
### 3.2 Flood Fill
> retval, image, mask, rect = cv.floodFill(image, mask, seedPoint, newVal[, loDiff[, upDiff[, flags]]])

### 3.3 Watershed
> markers = cv.watershed(image, markers)

### 3.4 Grabcuts
> mask, bgdModel, fgdModel = cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount[, mode])

### 3.5 Mean Shift
> dst = cv.pyrMeanShiftFiltering(src, sp, sr[, dst[, maxLevel[, termcrit]]])

### 3.6 Background Subtractor
* Frame Difference
* Running Average Background
* Gaussian Background
* Codebook
* ...

## 4. Filtrage d'image
__Flou est un filtre passe-bas__. Un filtre passe-bas est un filtre qui laisse passer les basses fréquences et qui atténue les hautes fréquences.
### 4.1 Flou moyen
> blur(InputArray __src__, OutputArray __dst__, Size __ksize__)
### 4.2 Flou de la boîte
> boxFilter(InputArray __src__, OutputArray __dst__, int __ddepth__, Size __ksize__)
### 4.3 Flou gaussien
> GaussianBlur(InputArray __src__, OutputArray __dst__, Size __ksize__, double __sigmaX__, double __sigmaY__=0)
### 4.4 Flou médian
> medianBlur(InputArray __src__, OutputArray __dst__, int __ksize__)
### 4.5 Flou bilatéral
> bilateralFilter(InputArray __src__, OutputArray __dst__, int __d__, double __sigmaColor__, double __sigmaSpace__)

## 5. Détection de contours
### 5.1 Filtre de Sobel
Il combine le flou gaussien et la dérivation différentielle. Il est apprécié pour sa simplicité et sa rapidité d'exécution. Ces qualité posent des problèmes lorsqu'il s'agit de traiter une image complexe.
> Sobel(InputArray __src__, OutputArray __dst__, int __ddepth__, int __xorder__, int __yorder__, int __ksize__=3, double __scale__=1, double __delta__=0, int __borderType__=BORDER_DEFAULT)

Gx = [-1 0 1; -2 0 2; -1 0 1]  
Gy = [-1 -2 -1; 0 0 0; 1 2 1]
### 5.2 Filtre de Scharr
Il est même rapide que le filtre de Sobel mais plus fort.  
Gx = [-3 0 3; -10 0 10; -3 0 3]  
Gy = [-3 -10 -3; 0 0 0; 3 10 3]
### 5.3 Filtre laplacien
Laplacian()
### 5.4 Filtre de Canny (MEILLEUR!)
{: .gitlab-red}
Il est bâti autour du filtre de Sobel pour améliorer.  
* Les filtres triangulaires utilisés par Sobel étant peu efficaces face à une image fortement bruitée, un filtre gaussien est utilisé.  
* Elle permet d'éliminer des faux contours.
### 5.5 Filtre dddd