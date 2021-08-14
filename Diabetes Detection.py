#Make sure all libraries are install in system through Command prompt
#Use command prompt to access this webapp on web browser through this command streamlit run "path of file"
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import image
import streamlit as st

# create a title and sub-title
st.write("""
# Diabetes Detection
Detect if someone has diabetes using Machine Learning and Python !
""")
#display an image
image = image.open('D:/python project/machine learning projects/application in python/images.jpg')
st.image(image, caption='ML', use_column_width=True)

#get the data
df = pd.read_csv('D:/python project/machine learning projects/application in python/diabetes.csv')

#set a sub header
st.subheader('Data Information')

#show data in table
st.dataframe(df)

#show statistics of the data
st.write(df.describe())

#show the data as chart
chart = st.bar_chart(df)

#split the data into independent x and dependent y variables
x = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values

#split the dataset into 75% training and 25 testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#get the feature input from user
def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    Blood_Pressure = st.sidebar.slider('Blood_Pressure', 0, 122, 72)
    Skin_Thickness = st.sidebar.slider('Skin_Thickness', 0, 93, 23)
    insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.5)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    diabetes_Pedigree_Function = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    Age = st.sidebar.slider('Age', 21, 81, 29)

    #store a dictionary
    user_data = {'Pregnancies': pregnancies,
             'Glucose': glucose,
             'Blood_Pressure': Blood_Pressure,
             'Skin_Thickness': Skin_Thickness,
             'insulin': insulin,
             'BMI': bmi,
             'DPF': diabetes_Pedigree_Function,
             'Age': Age
            }

    #transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features

#store the user input into a variable
user_input=get_user_input()

#set a subheader and display the users input
st.subheader('User Input : ')
st.write(user_input)

#create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(x_train,y_train)

#show the models metrics
st.subheader('Model Test Accuracy Score : ')
st.write(str(accuracy_score(y_test,RandomForestClassifier.predict(x_test))*100)+'%')

#store the models predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

#set a subholder and the classification
st.subheader('Classification')
st.write(prediction)