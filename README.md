# MLproject-Cardiotocography
<h1> Fetal Distress Prediction Based on Cardiotocographic (CTG) Data </h1> 
<h3> Members </h3> 
Suyashi Singhal, Harshita Gupta, Ayush Mahant, Rasagya Shokeen
Department of Computer Science
Indraprastha Institute of Information Technology, Delhi
{suyashi19478, harshita19467, ayush19353, rasagya19088}@iiitd.ac.in
<h3> Abstract </h3> 
Cardiotocography(CTG) is a monitoring technique used to
determine a fetus’s healthy being by simultaneously recording
the fetal heart rate and the mother’s uterine contractions.
The recorded CTG data can provide obstetricians with vital
information that can determine the well-being of the fetus and
the mother. However, visual inspection of such data might not be
very reliable, and hence we require additional ways of assessing
and evaluating fetal well-being. Advanced machine learning
algorithms should help us analyse the CTG data and predict
the fetal state. Our main aim is to develop a machine learning
model that can identify high-risk fetuses accurately comparable
to highly trained medical professionals. We hope that this
would play a significant role in reducing fetal mortality and
congenital disabilities globally. We have used a CTG dataset
containing 21 features and 2112 data points obtained from the
UCI Machine Learning Repository. We employ various robust
machine learning models to classify the fetal state into three
classes: Normal, Suspect, and Pathologic. We also perform
ten class classification of data based on Fetal Heart Rate
patterns. Trained on 3-fold cross-validation, we have analysed the classifiers’ successes using a variety of performance
metrics calculated from the confusion matrix. According to
our analysis, almost all the machine learning models have
shown satisfactory performance; however, in case of the 3-class
distribution random forest and bagging have proven to be the
most efficient in classification with an accuracy of 97.12% and
95.30%. Likewise for the 10-class distribution bagging and
random forest with an accuracy of 86.75% and 88.01% have
proven to be the most efficient. Furthermore, to improve the
accuracy of the models, we have used techniques like random
oversampling, principal component analysis, and extensive data
visualisation. The code and trained models are available at
https://github.com/suyashi912/MLproject-Cardiotocography

<h3> Introduction </h3> 
The number of fetal (unborn baby in the mother’s womb)
and maternal deaths every year worldwide is staggering. Undetected fetal abnormalities can progressively worsen, leading to
permanent damage to the fetus and even death. However, early
intervention can potentially be life-saving for both the mother as
well as the child.
<br> <br> 
Cardiotocography is one such technique that can concurrently
monitor the fetal heart rate (FHR) and uterine contractions(UC).
It can detect abnormal fetal state and movements, which could
be early symptoms of dangerous fetal distress conditions like
intrapartum hypoxia/asphyxia. If undiagnosed, this oxygen
deficiency can alter fetal physiology, resulting in probable brain
damage and even fetal death. A cardiogram is a machine used to
perform cardiotocography. It is an electronic fetal monitor that
employs the Doppler ultrasound effect using two transducers
placed externally on the pregnant woman’s abdominal wall
to record the FHR and UC signals. A time-scaled running
line graph depicts the intrauterine pressure measured from the
abdominal wall tensions. Trained clinicians and obstetricians can
interpret the CTG data. However, this method has the drawback
that visual inspection of the data is often unreliable and can vary
from person to person leading to inconsistent interpretations.
Over 50% of fetal deaths are due to this inconsistency in pattern
recognition and failure in receiving a timely intervention.
<br> <br> 

Therefore, integrating computerised machine learning methods with obstetrician interpretations can prove indispensable in predicting fetal distress conditions and providing early
intervention. In this study, our objective is to employ such
machine learning classifiers that can provide good performance
over unseen data and improve the cardiotocography technique.
We use a dataset obtained from the UCI Machine Learning
Repository. After performing appropriate feature selection and
analysis, we select 21 core features that influence the fetal state.
We thus have 2112 data samples out of the 2126 data points after
data preprocessing. We compare and contrast various machine
learning classifiers using performance and evaluation metrics
like accuracy, precision, recall and f1 score. Furthermore, we
use GridSearchCV to pick the best hyperparameters for each
model. It is worth mentioning that the imbalanced distribution of
data can prove to be a problem in classification. Thus, we utilize
a novel approach of random oversampling to increase the data
samples of minority classes and employ principal component
analysis for feature selection. In addition to the 3-class classification, we have also included the 10-class classification of the
data which is unseen in the papers analysed by us.
<br> <br> 
This report summarises the related works of literature we
have read, the machine learning models trained by us, and the
consequent analysis and results obtained from the model. We
also draw conclusions based on our study

<h3> Literature Survey </h3> 
Hoodbhoy et al. [1] study the precision of the machine
learning algorithm technique on the CTG dataset, aiming to
identify high-risk fetuses as accurately as highly trained medical
professions. For data balancing SMOTE technique is used that
avoids overfitting on skewed classes. Out of the ten machine
learning models, the XGBoost method had the highest overall
accuracy of about 93%. However, it did have the drawback of
having a comparatively low sensitivity in the "Suspect" state
compared to the "Pathologic" condition.<br> <br> 
Zhang et al. [3] present a novel approach for distinguishing fetal states into normal and pathological categories instead of
classifying the fetal states into three classes. This paper uses the
CTG dataset from UCI with 1831 groups, including 21 features
and one label. Principle Component Analysis (PCA) performs
dimensionality reduction and feature selection, improving
accuracy and computational time. It struggles with the problem
of outliers as the dataset used has no outlier values, which is not
justifiable in the real world. [3]<br> <br> 
In this paper[2], the authors present both R and Python
machine learning techniques for performance analysis. The
study shows that classification studies should be accountable to
models and parameter settings and the tools used. Four different
types of feature selection based on feature correlations and
various models are employed for this study.<br> <br> 
Unlike other studies, the authors in paper [4] also evaluated extreme learning machines[ ELM] algorithm with five
different activation functions apart from Random Forest Classifier, Support Vector Machines, Artificial Neural Network, and Radial Based function network.
<br> <br> 
Subha et al. select features using techniques like Gain Ration Attribute Evaluation, Relief Attribute Evaluation, and
Symmetrical Uncertainty Attribute Evaluation to obtain a dataset
with reduced features and thus increase model accuracy.[5]

<h3> Data Preprocessing and Visualisation </h3> 

<h4>  Dataset Description </h4> 
We have used the Carditocography raw data from the UCI Machine Learning Repository (SisPorto). The data consists of 2126
data samples and 28 features. It gives two types of classifications
: with respect to a morphologic pattern(10 classes) and to a fetal
state(3 classes). We first classify according to the fetal state : N,
S, P and use 21 features after performing feature selection. <br> <br> 
We then do the classification according to the Fetal Heart Rate
patterns into 10 classes : A, B, C, D, SH, AD, DE, LD, FS, SUSP.
<h4> Data Visualisation</h4> 
We plotted the correlation heat map (Figure 1) of the features
and selected the highly correlated ones. We also plotted a
boxplot (Figure 2) of the features to visualize distribution
of a variable using a five number summary which contains
the minimum value, first quartile(25%), median(50%), third
quartile(75%) and maximum value.
<h4> Preprocessing</h4> 
<h5>  Feature selection and cleaning</h5> 
We have dropped 7 features namely: Filename, Date, Segmentation File, b (Start time), e (End time), LBE(baseline value
- medical expert) and DR(repetitive decelerations) since these
features did not seem relevant to the fetal state. Furthermore we
dropped the rows with null values and removed duplicate data
samples. We thus obtained 2112 data samples and 21 features.
<h5>  Dimensionality Reduction (PCA) </h5> 
We performed dimensionality reduction using the Principal Component Analysis techniques on various kernels to reduce the dimension of the data. We noticed that our choice of 21 features
was appropriate. Interestingly, we observed that each kernel created two distinct groups of "Pathologic" class : one is close to
"Suspect" and one is far from it. However, the data is not linearly
separable as observed from the plots of various kernels. We have
added the plot for the "Linear" kernel (Figure 3) for reference.
All other kernels give similar plots.

<h5>Normalization</h5> 
In order to give equal weights to each feature in the dataset so
that no single variable steers model performance in one direction,
we performed data normalization. We have used the Min Max
Scaling method that squishes the feature values between 0 and 1.

<h5>Oversampling data</h5> 
The division of 2112 samples into 3 classes is as follows: 1646
normal, 292 suspect and 174 pathologic. This indicates that the
dataset is imbalanced across the given classes. Thus, we used
the random oversampling technique to avoid overfitting of the machine learning model on skewed classes by increasing the
data samples of the classes with minority instances. It randomly
selects minority class samples and randomly adds them to the
dataset to balance it. We got 3588 data samples after oversampling. 

<h3>Methodology</h3> 
<h4>Model Details</h4> 
We have split the above dataset into a training and testing set
using a 70:30 stratified split. After that, we trained the data on
3 fold cross-validation. Hence we achieved a train:validation:
test split of 47:23:30. After dividing the dataset, we chose some
supervised learning models to train and test on the dataset. We
also performed hyperparameter tuning using GridSearchCV and
chose the best model for training and testing. The description of
the different models that we employed are as follows:
1. Logistic Regression: It is a statistical model that uses a
logistic function to model a binary dependent variable in its
basic form. We have used the multinomial logistic regression to perform multi-class classification by changing the
loss function to cross-entropy loss and predict probability
distribution to a multinomial probability distribution.
2. Naive Bayes: It uses probability theory to classify data.
Naive Bayes classifier algorithms make use of Bayes’
theorem. All attributes of a data point under study are
considered to be independent of each other.
3. Decision Trees: It is a non parameterized supervised
learning technique with a pre-defined target variable and is
often used in classification problems.
4. Random forest: It is an ensemble learning method where
many decision trees are constructed at training time. The
output of the random forest is the class selected by the
majority of the trees.
5. K - Nearest Neighbors: It is based on a supervised learning
technique that calculates the nearest k neighbours for each
data point. The majority label among those data points is
returned as the predicted class.
6. Boosting: Boosting is an ensemble modelling method
which combines multiple weak learners to make a strong
learner based on weighted averages.
7. Bagging: It is a type of ensemble learning technique, also
called bootstrap aggregation. It involves randomly selecting
data with replacement in order to achieve reduced variance.
8. Support Vector Machine: It classifies the given labelled
training data by creating an optimal hyperplane for classification. It can be divided based on kernel type like Linear,
Polynomial, Sigmoid and Radial Basis Function(RBF).
9. Multi Layer Perceptron: It is a type of feedforward Artificial Neural Network(ANN) consisting of input layer, output
layer and hidden layers. It uses the back propagation algorithm to classify instances of data.
<h4> Performance metrics</h4> 
We used accuracy, precision, recall and F1-score as the evaluation metrics to test our models :
1. Accuracy: Accuracy measures the overall efficiency of a
classifier. We require that most of the fetal states are classified correctly for a good performance. It is worth noting that
even predicting normal state correctly is important because
wrong predictions lead to more cesarean sections.
Accuracy = TP / ( TN +FP+ FN +TP)
2. Precision: It is the ratio of true positives to the total of the
true positives and false positives. Here, it is the measure of
the number of fetal states classified correctly out of all the
positive samples.
Precision = TP / (TP + FP)
3. Recall: It is the ability of a classifier to categorize positively
labeled data. Hence, for all fetuses, it tells us how many we
correctly classified.
Recall = TP / (TP + FN)
4. F1 score: It is the harmonic mean between precision and
recall. High F1 score enables us to get a good trade-off between precision and recall.
F1-Score = TP / (TP + 0.5 (FP + FN))
<br> <br> 
<h3> Results </h3> 
The experimental results of the machine learning models on
the testing data are depicted in the given table and histogram. The
precision, recall and F1 score metrics are the weighted averages
of the corresponding scores of the classes.
<br> <br> 
The highest accuracy for the 3 class classification with
a similar performance of the classification algorithms has
been observed both in Bagging (97.49%) and Random Forest
(97.21%) Apart from that Support Vector Machine(96.37%),
KNN(95.26%) and Decision Trees(94.61%) also have a considerably good accuracy. On the other hand Gaussian Naive Bayes
gives the least accuracy (73.14%).
<br> <br> 
Logistic Regression gives a decent performance as compared to the other models. We have used the multinomial logistic
regression to perform multi-class classification, which offers better performance over the One-vs-All method. AdaBoost, a
boosting algorithm based on decision trees also works decently
in determining the fetal distress. Gaussian Naive Bayes has
the worst accuracy since it assumes no dependency between
attributes which is not the case as the heat map depicts a high
correlation between various features.
<br> <br> 
The K-Nearest Neighbours model works well because it
operates on the correlation of features. We have correlated
features like Minimum and Maximum values of FHR signals
and several other derived variables related to the FHR histogram,
which work well with the KNN machine learning model. Furthermore, the KNN model favors a noiseless dataset, and the fact
that we have a clean dataset enhances its predictions. Similarly
Decision Trees and SVM have good accuracies. The decision
tree can easily handle high dimensional non parameterized data
and works well with non-linearly separable patterns. Hence, it
performs well after pruning and gives an accuracy of 94.61%.
SVM based on kernel functions gives an accuracy of 96.37%
<br> <br> 
Multi Layer Perceptron uses the concept of Artificial Neural
Network to train a robust model having an accuracy of 94.88%.
The activation function employed here is logistic.
<br> <br> 
Bagging and Random Forests give the best accuracy and
precision. Random forest enhances the performance of decision
trees further. It is an ensemble method that combines the output
of multiple unpruned decision trees and makes a prediction
based on the majority vote. It gives an accuracy of over 97.21%
and achieves similar precision and recall scores as well. Bagging
or Bootstrap Aggregation uses the ensemble learning method
and combines various weak decision tree based classifiers into
a strong classifier and reduces variance to give an accuracy of
97.49%.
<br> <br> 
We observe that though many of these models are successful in predicting the Normal and Pathological state accurately,
there is a considerable drop in the values of the evaluation
metrics while predicting the suspect state of the fetus. This can
prove to be problematic in some cases as even the suspect state
might be a signal of fetal distress. 
<br> <br> 
The highest accuracy for the 10 class classification with a
similar performance of the classification algorithms has been observed both in Bagging (86.75%) and Random Forest (88.01%)
Classifiers followed by Multilayer Perceptron (84.54%). Apart
from that the Support Vector Machine(82.33%)also gives a
pretty good accuracy. In contrast, Gaussian Naive Bayes gives
the least accuracy (13.09%).
<br> <br> 

Random Forest gives the best metrics including high precision value while predicting the state using fetal heart rate patterns.
Bagging also gives a similar accuracy of 86.75% which suggests that the random classifier based approach combined with
bagging / bootstrap aggregation can be utilized for an improved
performance.
<br> <br> 
Naive Bayes performs poorly since it is based on an oftenfaulty assumption that features are independent of each other.
This results in biased prediction of the posterior probabilities,
thus giving inaccurate results, particularly for multi-class classification.
<br> <br> 
Multi Layer Perceptron makes use of the hidden layers and
back propagation algorithm to give a good accuracy of 84.54%.
Support Vector Machine and Logistic Regression are also
successful in giving >80% prediction metric values.
<h3> Conclusion </h3> 
Through this study it is shown that pre-processing, extensive
machine learning model techniques, and difference in tools used
play a vital role in classification scores analysis. The machine
learning model techniques used are Multinomial Logistic regression, Gaussian Naive Bayes, Decision trees, Random forests,
K-Nearest Neighbors, Support Vector Machine, AdaBoost,
Bagging and Multi Layer Perceptron with 3-fold cross validation. In both the cases of three as well as ten class distribution,
Random Forests and Bagging gave the best accuracy. Our
methodology not just includes data cleaning and normalization
but also incorporates advanced feature selection and engineering
methods like Principal Component Analysis used for Dimensionality Reduction. Finally, we have selected the relevant
features and observed that the data is not linearly separable even
after performing dimensionality reduction. We have achieved
almost perfect accuracy, precision, recall, and F1 score metrics
of around 95-98% in 6 out of 9 models in case of the 3-class
prediction and metrics of around 80-90% for 10-class prediction,
which reiterates that the feature engineering performed by us has
performed well. CTG requires expert interpretation unavailable
in remote areas, leading to unavailability of dataset from women
in such areas, which leads to a problem in identification of proper
set of classification of normal, suspect, and pathological cases in
India. The dataset is obtained from the repository of a developed
country which only took into consideration a particular segment
of population. Hence the dataset does not consider differences in
socio-demographic characteristics of pregnant women and some
other relevant features like age, nutritional status and so on.This
is one of the drawbacks of this study.
<h3> References </h3> 
<ol> 
<li> 
Z. Hoodbhoy, Md. Noman, A. Shafique, Ali Nasim, D.
Chowdhury, B. Hasan, Use of Machine Learning Algorithms
for Prediction of Fetal Risk using Cardiotocographic Data,
2019
</li>
<li> 
Y. Zhang, Z. Zhao, Fetal State Assessment Based on Cardiotocography Parameters Using PCA and AdaBoost, 2017
</li>
<li> 
S.C.R Nandipati, C. XinYing, Classification and Feature
Selection Approaches for Cardiotocography by Machine
Learning Techniques, 2020
</li>
<li> 
Z.Cömerta, A.F. Kocamazb, Comparison of Machine
Learning Techniques for Fetal Heart Rate Classification,
2016
</li>
<li> 
V.Subha, Dr.D.Murugan, Jency Rani, Dr.K.Rajalakshmi ,
Comparative Analysis of Classification Techniques using
Cardiotocography Dataset, 2013
</li>
</ol> 














