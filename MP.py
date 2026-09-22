import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Data Collection
df = pd.read_csv("accidents.csv")
print("============= Original Dataset: =============\n")
print(df)

# Data Cleaning
print("============= Missing Values: ==========\n")
print(df.isnull().sum())

print("============= Duplicate Records: =============\n")
print("Values is:",df.duplicated().sum())

# Data Preprocessing
weather_encoder = LabelEncoder()
road_encoder = LabelEncoder()
severity_encoder = LabelEncoder()

df['Weather'] = weather_encoder.fit_transform(df['Weather'])
df['Road_Type'] = road_encoder.fit_transform(df['Road_Type'])
df['Severity'] = severity_encoder.fit_transform(df['Severity'])

# Exploratory Data Analysis
print("============= First Five Rows:=============\n")
print(df.head())

print("============= Last Five Rows: =============\n")
print(df.tail())

print("========== Statistical Summary: ==========\n")
print(df.describe())

# Data Visualization
sns.countplot(x='Weather', data=df)
plt.title("Weather Wise Accidents")
plt.show()

sns.countplot(x='Road_Type',data=df, color = "green")
plt.title("Road Type Wise Accidents")
plt.show()

sns.countplot(x='Severity', data=df , color = "red")
plt.title("Accident Severity Distribution")
plt.show()

# Model Building
X = df[['Weather','Road_Type','Speed_Limit','Vehicle_Count']]
y = df['Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Severity Prediction
y_pred = model.predict(X_test)
new_data = pd.DataFrame(
    [[2,1,75,3]],
    columns=['Weather','Road_Type','Speed_Limit','Vehicle_Count']
)

prediction = model.predict(new_data)
result = severity_encoder.inverse_transform(prediction)
print("=============================================")
print("Predicted Severity =", result[0])

# Result Analysis
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy =", accuracy)
print(classification_report(y_test, y_pred))
print("=============================================")