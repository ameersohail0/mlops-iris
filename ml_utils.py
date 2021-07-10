from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# define a Gaussain NB classifier
gnb_clf = GaussianNB()

# creating an alternative classifier
# using a logistic regression classifier
logreg_clf = LogisticRegression(penalty='l2',C=1.0, max_iter=10000)

# classifier with best accuracy will be used
clf = None

# define the class encodings and reverse encodings
classes = {0: "Iris Setosa", 1: "Iris Versicolour", 2: "Iris Virginica"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def load_model():
    # load the dataset from the official sklearn datasets
    X, y = datasets.load_iris(return_X_y=True)

    # do the test-train split and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Training Guassain NB Classifier
    gnb_clf.fit(X_train, y_train)

    # Training Logistic Regression Classifier
    logreg_clf.fit(X_train, y_train)

    # calculate the accuracy score of Guassain NB Classifier
    gnb_acc = accuracy_score(y_test, gnb_clf.predict(X_test))

    # calculate the accuracy score of Logistic Regression Classifier
    logreg_acc = accuracy_score(y_test, logreg_clf.predict(X_test))

    # modifying the global classifier
    global clf
    # Using the model with best accuracy
    if (logreg_acc >= gnb_acc):
        clf = logreg_clf
        print(f"Logistic Regression Model trained with accuracy: {round(logreg_acc, 3)}")
    else :
        clf = gnb_clf
        print(f"Guassain NB Model trained with accuracy: {round(gnb_acc, 3)}")


# function to predict the flower using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    print(f"Model prediction: {classes[prediction]}")
    return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)
    print("Model has trained with the new dataset provided from the feedback loop")
