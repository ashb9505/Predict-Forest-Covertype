# FOREST COVERTYPE PREDICTION USING MACHINE LEARNING 




# Problem Statement
Develop a machine learning model to accurately predict the forest cover type based on given geographic and environmental attributes.




# Data and Model Description


## Dataset
Size: 581,012 observations


Features (54 features)


    Numerical Variables: 
        Elevation: Height above sea level 
        Aspect: Direction of slope faces
        Slope: Slope steepness
        Horizontal and vertical distance to Hydrology
        Horizontal and vertical distance to Roadway
        Horizontal and vertical distance to Fire Points
        Hillshade indexes: how much sunlight is blocked
    
    Categorical Variables (One hot encoder):
        Soil type (40 types)
        Wilderness Area (4 types)
    
    Target Variable: Forest cover type (7 classes)


This is a Multi-classification problem.


## Model Selection


Logistic Regression: Baseline classifier.


k-Nearest Neighbor: Utilizes proximity for classification.


Decision Tree & Random Forest: Effective for complex relationships.

AdaBoost: Support Random Forest for a better model


# Analysis Strategy
## Exploratory Data Analysis (EDA)


Visualize feature distributions and target distribution.


Since data is highly imbalanced, we need to choose a proper scaling and model to work on this problem. 


## Data Preprocessing
    
Data Cleaning: No missing values, no duplicates.


Data Encoding: Categorical variables are in one hot encoder, no further action needed. 


Data Scaling: Standard Scaler.


## Approach for Imbalanced Data


Use SMOTE for data transform to balance minority classes.


Use the Overfitting Resampling method.


Remove irrelevant/less related features.


Find Hyperparameters.


# Model Training and Evaluation
Function split_and_scale that randomly shuffle data, split 70/30 Training/Testing with Stratified Sampling and Standard Scaler
    
    def split_and_scale(df):
    df = df.copy()


    np.random.seed(0)


    # Split df into X and y
    y = df['CoverType'].copy()
    X = df.drop('CoverType', axis=1).copy()


    # shuffle the data
    shuffle = np.random.permutation(np.arange(X.shape[0]))
    X, y = X.iloc[shuffle], y.iloc[shuffle]


    # Train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, stratify=y, random_state=42)


    # Scale X
    scaler = StandardScaler()
    scaler.fit(X_train)


    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


    return X_train_scaled, X_test_scaled, y_train, y_test


Function split_and_scale_SMOTE with additional SMOTE


    # Apply SMOTE to the training set
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)


Function evaluate_model that generates confusion matrix and classification report
   
    def evaluate_model(model, class_balance, X_test, y_test):


    model_acc = model.score(X_test, y_test)
    print("Accuracy ({}): {:.2f}%".format(class_balance, model_acc * 100))


    y_pred = model.predict(X_test)


    cm = confusion_matrix(y_test, y_pred)
    clr = classification_report(y_test, y_pred)


    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cbar=False, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


    print("Classification Report:\n----------------------\n", clr)


## Model 1: Logistic Regression
Compare LR results in four cases: Standard Scaling, Standard Scaling with SMOTE, Balanced class weight, and Overfitting Resampling.


Set max_iteration = 1000


With Logistic Regression, Standard scaling with/without SMOTE has the same overall accuracy score (72%), and tends to not perform well on minority classes (especially class 4 and 5 with precision score of 0.24/0.29). With overfitting resampling, the overall accuracy score and precision score for the majority class decreases slightly. However, the model works better with minority classes (precision score .72/0.62 for class 4 and 5). 


Balanced class weight does not work well in this case. It only achieves an accuracy score of 60%. 


Time needed to run logistic regression is fast (around 3-5 minutes per model). 


## Model 2: k-Nearest Neighbor
Compare kNN result in two cases: Standard Scaling and PCA


Set n_neighbor = 5


With k-Nearest Neighbor, Standard Scaling has a better accuracy score compared to the best model of Logistic Regression (92%). The precision, recall and f1-score also outperforms LR. With the effort to optimize the result, PCA was used to optimize the model. However, the accuracy is pretty low compared to kNN with Standard Scaling (89%). 


Time needed to run kNN is average (around 10-15 minutes per model)


## Model 3: Decision Tree


The model demonstrates slightly superior performance compared to kNN, achieving an accuracy of 93% with commendable precision scores. The majority of values are predicted accurately, with the lowest score recorded at 82%.


## Model 4: Random Forest


The Random Forest model achieved an impressive overall accuracy of 95.15%, outperforming previous models. The removal of irrelevant features by setting a stricter threshold (< 0.0005) resulted in a slight improvement to 95.20%. Further refinement using hyperparameters (specifically, n_estimators = 200) saw a modest increase to 95.25% accuracy. Additionally, the precision score for class 3 rose to 0.90.


## Model 5: AdaBoost


Despite utilizing the dataset post-SMOTE to address sensitivity to overfitting and data imbalance, the achieved accuracy plateaued at 95.10%. Cannot outperform random forest.




# Conclusion 
Accuracy: Logistic Regression < kNN < Decision Tree < AdaBoost < Random Forest


Decision Tree and Random Forest tend to work better with imbalanced data. However, the time it takes to get the result from these two models is significantly longer than kNN and Logistic Regression, so we should choose the appropriate method based on our time and purposes.


Future work: Emphasize more on variables that have higher importance based on Decision Tree/Random Forest/AdaBoost result and generate models based on them. 


