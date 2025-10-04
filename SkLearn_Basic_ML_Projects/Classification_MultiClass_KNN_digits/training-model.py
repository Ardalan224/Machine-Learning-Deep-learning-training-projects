from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

digits = load_digits()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size= 0.3, random_state = 42)

knn = KNeighborsClassifier(n_neighbors= 3)

knn.fit(x_train, y_train)



# Evaluate model
print("Model accuracy:", knn.score(x_test, y_test))

# Save model
joblib.dump(knn, "digits_KNN_model.pkl")
print("Model saved as digits_KNN_model.pkl")