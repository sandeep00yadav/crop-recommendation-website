import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv("Crop_recommendation.csv")
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = RandomForestClassifier()
model.fit(X_train,y_train)
prediction = model.predict(X_test)
accuracy=accuracy_score(y_test,prediction)
print("Accuracy:",accuracy)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f )
    print("Model saved as model.pkl")



