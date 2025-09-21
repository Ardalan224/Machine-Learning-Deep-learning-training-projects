from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib


digits = load_digits()
model = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size= 0.2)

model.fit(x_train, y_train)

# Evaluate model
print("Model accuracy:", model.score(x_test, y_test))

# Save model
joblib.dump(model, "digits_model.pkl")
print("Model saved as digits_model.pkl")


