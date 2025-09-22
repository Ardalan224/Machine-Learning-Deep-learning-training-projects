from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import joblib

digits = load_digits()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state= 42)

model = RandomForestClassifier(n_estimators= 40)
model.fit(x_train, y_train)

print("Accuracy score is: ", model.score(x_test, y_test))

# Save model
joblib.dump(model, "digits_RandomForest_model.pkl")
print("Model saved as digits_RandomForest_model.pkl")