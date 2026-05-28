import pandas as pd
from matplotlib import pyplot as plt


from sklearn.datasets import load_digits

digits = load_digits()

# Check available attributes
print(dir(digits))

# First data sample
print(digits.data[0])

# Display first 5 images
plt.gray()

for i in range(5):
    plt.matshow(digits.images[i])

# First 5 targets
print(digits.target[0:5])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    digits.data,
    digits.target,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LogisticRegression

# Increase iterations
model = LogisticRegression(max_iter=10000)

# Train the model
model.fit(x_train, y_train)

# Accuracy score
print(model.score(x_test, y_test))