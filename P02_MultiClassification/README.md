# Animal Multi-classification | Multi-classification des animaux

[English](#0-Introduction)  
[Français](#0-Introduction-fr)


### 0. Introduction
This project uses 3 datasets to do 3 tasks:
* Stage 1: classification of animal classes - to predict that the target is a mammal or bird
* Stage 2: classification of species - to predict that the target is a rabbit, rat or chick
* Stage 3: multi-classification - to predict the class and the species at the same time

### 1. Structure of project
A. Datasets
  * Structure of dataset: number of images
      * train
        * chicken: 310
        * rabbits: 310
        * rats: 270
        
      * val
        * chicken: 30
        * rabbits: 30
        * rats: 20
        
    * Images-rename.py: rename images in bulk
    
B. Stage_1 Classes_classification
  * Classes_make_anno.py: generate annotations for train and test dataset; choose rabbits and chicken as datasets and predict whether the image shows a mammal or bird (0, 1)

  * Classes_Network.py: network for classes classification

  * Classes_classification.py: main file for training model and evaluation and calling trained model to do prediction, including result visualization

  * Classes_train_annotation.csv/Classes_val_annotation.csv: annotations for classes classification (generated by Classes_make_anno.py)

C. Stage_2 Species_classification
  * Species_make_anno.py: generate species annotations for train and test dataset of rabbits, rats and chicken (0, 1, 2)
  
  * Species_Network.py: network for species classification
  
  * Species_classification.py: main file for training model and evaluation and calling trained model to do prediction, including result visualization
  
  * Species_train_annotation.csv/Species_val_annotation.csv: annotations for species classification (generated by Species_make_anno.py)
  
D. Stage_3 Multi-classification
  * Multi_make_anno.py: generate both classes and species annotations for train and test dataset (0,0; 0,1; 0,2; 1,0; 1,1; 1,2)
    
  * Multi_Network.py: network for simultaneous classes and species classification
    
  * Multi_classification.py: main file for training model and evaluation and calling trained model to do prediction, including result visualization
    
  * Multi_train_annotation.csv/Multi_val_annotation.csv: annotations for classes and species classification (generated by Multi_make_anno.py)

### 2. Approach
##### A. Data preprocessing
Many deep learning applications require us to build the datasets of our own. In this project, these images of training and validation/test are grabbed from the Internet.

In Stage 1, mammal and bird data are labeled as 0 and 1.  
In Stage 2, rabbits, rats and chicken are labelled as 0, 1 and 2.

##### B. Data loading
Using ___torchvision.transforms___ and ___torch.utils.data.DataLoader___ to load train data and test data.

Then define the data in different ways.  
In Stage 1:
> sample = {'image':image, 'classes':label_classes}

In Stage 2:
> sample = {'image':image, 'species':label_species}

In Stage 3:
> sample = {'image':image, 'classes':label_classes, 'species':label_species}

##### C. Data validation
Goal: validate the dataset and ensure that the images are correctly labeled.  
Method: display randomly an image and its label in the dataset before training a model.

##### D. Network building
Choose carefully the layers and their dimensions, and put them in appropriate positions.  
Stage 1 and Stage 2 are tasks of single classification, we can put a ___Softmax___ classifier behind the FC layer.  
Stage 3 is a task of multi-classification, we can connect 2 ___Softmax___ classifiers behind the FC layer, one is for classes classification and the other is for species classification.

##### E. Model training and testing
Train the model in train dataset and evaluate the model in test dataset, record separately the losses and accuracies, decide whether the training is effective and efficient, record the best model which has the highest accuracy.  
Stage 1 and Stage 2 are tasks of single classification, so the loss is of single classification.  
Stage 3 is a task of multi-classification, we can use linear weighted losses to calculate a final loss for training.  
Loss function: Cross Entropy Loss  
Optimization functions: SGD, Momentum, Adam, etc.

##### F. Parameter tuning
Tune the hyperparameters such as learning rate, momentum, weight of loss to optimize the result.

##### G. Data visualization
Predict images with the trained model so as to visualize directly the effect of the predictor.

### 3. Analysis

##### Environment of the project:
> OS: Windows 10  
> RAM: 16 GB  
> CPU: Intel Core i7-7700HQ 2.80GHz  
> GPU: Nvidia GeForce GTX 1050 __(PyTorch installed with CUDA enabled GPU)__   
> IDE: MS Visual Studio 2019 & VS Code  

##### Stage 1 and Stage 2: Manipulations
1. Create 3 networks:
    * The original network: conv, maxpool, relu, conv, maxpool, relu, fc, relu, dropout, fc, softmax
    * The original network + more layers: conv, maxpool, relu, conv, maxpool, relu, conv, maxpool, relu, fc, relu, dropout, fc, softmax
    * The original network + batch normalization
2. Do grid search on learning rate, optimization method(SGD, Adam), and regularization. Save the results in png.
3. All the numbers of epoches are set to 30.
4. Remove softmax in the original net classes, since ___CrossEntropyLoss = Softmax + NLLLoss(Negative Log Likelihood Loss)___ contains Softmax already.

##### Stage 1 and Stage 2: Findings
1. Best result when the regularization is at 0, which means no regularization. Maybe it is because the dropout layer is sufficient.
2. For SGD, lr=0.01 is quite good for both classes and species classifications. Batch normalization can help convergence very slightly.
3. For Adam, lr=1e-4 may be helpful but the advantage is not very obvious.
4. Both SGD and Adam give similar results at the end.

##### Stage 3: Manipulations
1. Use 1 network:
    * The original network in Stage 1: conv, maxpool, relu, conv, maxpool, relu, fc, relu, dropout, fc, softmax
2. Try two types of minimization, one with loss=loss1+loss2*wt, with wt=1.0 and 1.5 (loss1 is for classes and loss2 is for species). The other one trys num_epoches1 first to minimize loss2, then try to minimize loss1.

##### Stage 3: Findings
1. No matter which type of minization is used, the accuracies converge to about 70% for classes and about 50% for species. The two-step method might be a little better. 
2. Data augmentation is necessary if we want to improve the result. The current amount of images (<1000) is way too small.
3. It seems that weight=1.5 is a little instable.

---
### 0. Introduction fr
Ce projet utilise les 3 ensembles de données pour faire 3 tâches :
* Phase 1 : classification de classes des animaux - pour prédire que ce soit un mammifère ou un oiseau
* Phase 2 : classification d'espèces des animaux - pour prédire que ce soit un lapin, un rat ou un oiseau
* Phase 3 : multi-classification - pour prédire la classe et l'espèce en même temps

### 1. Structure du projet
A. Ensembles des données
  * Structure des ensembles des donnéess : nombre des images
      * train
        * poulets : 310
        * lapins : 310
        * rats : 270
        
      * val
        * poulets : 30
        * lapins : 30
        * rats : 20
        
    * Images-rename.py: renommer les images
    
B. Phase_1 Classes_classification
  * Classes_make_anno.py: générer les annotations pour les ensembles de données train et test; choisir lapins et poulets comme données et prédire si l'image affiche un mammifère ou un oiseau (0, 1)

  * Classes_Network.py: réseau pour classification de classes

  * Classes_classification.py: fichier principal pour le modèle de training et d'évaluation, appeler le modèle entraîné pour faire de la prédiction, y inclus la visualisation des résultats

  * Classes_train_annotation.csv/Classes_val_annotation.csv: annotations pour classification de classes (générées par Classes_make_anno.py)

C. Phase_2 Species_classification
  * Species_make_anno.py: générer les annotations d'espèces pour les ensembles de données train et test de lapins, rats et poulets (0, 1, 2)
  
  * Species_Network.py: réseau pour classification d'espèces
  
  * Species_classification.py: fichier principal pour le modèle de training et d'évaluation, appeler le modèle entraîné pour faire de la prédiction, y inclus la visualisation des résultats
  
  * Species_train_annotation.csv/Species_val_annotation.csv: annotations pour classification d'espèces (générées par Species_make_anno.py)
  
D. Phase_3 Multi-classification
  * Multi_make_anno.py: generate both classes and species annotations for train and test dataset (0,0; 0,1; 0,2; 1,0; 1,1; 1,2)
    
  * Multi_Network.py: network for simultaneous classes and species classification
    
  * Multi_classification.py: main file for training model and evaluation and calling trained model to do prediction, including result visualization
    
  * Multi_train_annotation.csv/Multi_val_annotation.csv: annotations for classes and species classification (generated by Multi_make_anno.py)

### 2. Approche
##### A. Prétraitement des données
Beaucoup d'applications Deep Learning nous obligent à créer nos propres ensembles de données. Dans ce projet, ces images pour entraînement et évaluation sont récupérées sur Internet.

Dans Phase 1, les données de mammifère et d'oiseau sont libellées avec 0 et 1.  
Dans Phase 2, les données de lapins, de rats et de poulets sont libellées avec 0, 1 and 2.

##### B. Chargement des données
Utiliser ___torchvision.transforms___ et ___torch.utils.data.DataLoader___ pour charger les données d'entraînement et de test.  

Ensuite, définir les données en différentes manières.  
Dans Phase 1 :
> sample = {'image':image, 'classes':label_classes}

Dans Phase 2 :
> sample = {'image':image, 'species':label_species}

Dans Phase 3 :
> sample = {'image':image, 'classes':label_classes, 'species':label_species}

##### C. Validation des données
But: valider les ensembles de données et assurer que les images sont correctement étiquetées.  
Méthode: afficher aléatoirement une image et son étiquette dans les ensembles de données avant d'entraîner un modèle.

##### D. Etablissement des modèles
Choisir soigneusement les couches et leurs dimensions et les placer dans les positions appropriées.  
Phase 1 et Phase 2 sont les tâches de classification unique, on peut placer un classificateur ___Softmax___ avant la couche FC.  
Phase 3 est une tâche de multi-classification, on peut prendre 2 classificateurs ___Softmax___ avant la couche FC dont un est pour la classification de classes et l'autre est pour la classification d'espèces.

##### E. Entraînement et test des modèles
Entraîner le modèle dans l'ensemble de données train et évaluer le modèle dans l'ensemble de données test, noter séparément les pertes et les justesses, décider si l'entraînement est efficace et efficient, enregistrer le meilleur modèle qui a la justesse la plus haute.  
Phase 1 et Phase 2 sont les tâches de classification unique, donc la perte est pour la classification unique.  
Phase 3 est une tâche de multi-classification, on peut utiliser les pertes linéairement pondérées afin de calculer une perte totale pour entraînement.  
Fonction de perte: Perte d'Entropie Croisée (Cross Entropy Loss)  
Functions d'optimisation: SGD, Momentum, Adam, etc.

##### F. Réglage des paramètres
Régler les hyperparamètres comme learning rate, momentum, poids de perte pour optimiser le résultat.

##### G. Visualisation des données
Prédire le contenu des images avec le modèle entraîné pour visualiser directement l'effet du prédicteur.

### 3. Analyse

##### Environnement du projet :
> OS : Windows 10  
> RAM : 16 GB  
> CPU : Intel Core i7-7700HQ 2.80GHz  
> GPU : Nvidia GeForce GTX 1050 __(PyTorch installé avec GPU compatible CUDA)__   
> IDE : MS Visual Studio 2019 & VS Code  

##### Phase 1 and Phase 2: Manipulations
1. Créer 3 réseaux:
    * Un réseau classique : conv, maxpool, relu, conv, maxpool, relu, fc, relu, dropout, fc, softmax
    * Le réseau ci-dessus + plus de couches : conv, maxpool, relu, conv, maxpool, relu, conv, maxpool, relu, fc, relu, dropout, fc, softmax
    * Le réseau ci-dessus + batch normalisation
2. Faire la recherche dans une grille (grid search) sur le taux d'apprentissage, la méthode d'optimisation (SGD, Adam), et la régularisation. Enregistrer les résultats en image.
3. Tous les époques sont fixées à 30.
4. Retirer softmax parce que ___CrossEntropyLoss = Softmax + NLLLoss(Negative Log Likelihood Loss)___ qui contient softmax déjà.

##### Phase 1 and Phase 2: Conclusions
1. Le meilleur résultat quand la régularisation est à 0, ce qui signifie non-régularisation. Peut-être c'est parce que la couche de dropout est déjà suffisante.
2. Pour SGD, lr=0.01 est assez bon pour les classifications de classes et d'espèces. La normalisation par lot peut aider la convergence légèrement.
3. Pour Adam, lr=1e-4 peut être utile mais l'intérêt n'est pas très évident.
4. SGD and Adam tous les deux donnent les résultats similaires à la fin.

##### Phase 3: Manipulations
1. Utiliser un réseau :
    * Le réseau classique dans Phase 1 : conv, maxpool, relu, conv, maxpool, relu, fc, relu, dropout, fc, softmax
2. Essayer deux types de minimisation dont un utilise loss=loss1+loss2*wt, avec wt=1.0 et 1.5 (loss1 est pour classes et loss2 est pour espèces). L'autre type essaie d'abord num_epoches1 pour minimiser loss2, et après essaie de minimiser loss1.

##### Phase 3: Conclusions
1. Quel que soit le type de minimisation utilisé, les justesses convergent vers 70% pour les classes et vers 50% pour les epsèces. Le type en 2 étapes pourrait être un peu meilleur.
2. L'augmentation des données est donc nécessaire si nous voulons améliorer le résultat. La quantité actuelle d'images (<1000) est trop petite.
3. Il semble que weight=1.5 rende le résultat un peu instable.