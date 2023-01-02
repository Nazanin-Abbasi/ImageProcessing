from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  classification_report, accuracy_score


class MachineLearning:
    def classification(self, dataset):
        X = dataset.loc[:, 'Contour Size':'Corner Size']
        y = dataset.loc[:, ['Label']]
        y = y.astype('int')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train, sample_weight=None, check_input=True, X_idx_sorted="deprecated")

        y_pred = clf.predict(X_test)

        result = classification_report(y_test, y_pred)
        print("Classification Report: ", result)

        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: ", accuracy)
