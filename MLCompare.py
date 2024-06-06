import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly_express as px
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from ydata_profiling import ProfileReport    
from streamlit_pandas_profiling import st_profile_report
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.impute import SimpleImputer
import pickle

df = None 


if os.path.exists('./dataset.csv'):
   df = pd.read_csv('dataset.csv', index_col=None)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]
    
    return df



with st.sidebar:
    st.image("1.png")
    st.title("MLCompare")
    choice = st.radio("Navigation", ["Home", "Upload", "Filter", "Perform EDA", "Profiling Report", "Data Preparing", "Modelling", "Analysis", "Download"])
    st.info("AutoML Platform for Comparative Analysis of Machine Learning Models.")
    st.info("Devloped by Ai-SSMS")


if choice == "Home":
    st.title("What is AutoML ?")
    st.video("AutoML.mp4")
    st.title("About the Platform ?")
    st.video("Platform.mp4")


if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        df.to_csv('dataset1.csv', index=None)
        st.dataframe(df)
    st.image("2.jpeg")

if choice == "Filter":
    filtered_df = filter_dataframe(df)
    st.write("Filtered DataFrame:")
    st.dataframe(filtered_df)
    filtered_df.to_csv('dataset.csv', index=None)
    st.success('The filtered DataFrame has been successfully saved to "dataset.csv"')


if choice == "Perform EDA":
    st.title("Exploratory Data Analysis")
    df = pd.read_csv('dataset.csv', index_col=None)

    eda_choise = st.selectbox('Pick the operation',['','Show shape','Show data type','Show messing values','Summary','Show columns','Show selected columns','Show Value Counts'])

    if eda_choise =='Show shape':
        st.write(df.shape)

    if eda_choise =='Show data type':
        st.write(df.dtypes)

    if eda_choise == 'Show messing values':
        st.write(df.isna().sum())

    if eda_choise =='Summary':
        st.write(df.describe())

    if eda_choise =='Show columns':
        all_columns = df.columns
        st.write(all_columns)

    if eda_choise =='Show selected columns':
        selected_columns = st.multiselect('Select desired columns',df.columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    if eda_choise =='Show Value Counts':
        try:
            selected_columns = st.multiselect('Select desired columns', df.columns)
            new_df = df[selected_columns]
            st.write(new_df.value_counts().rename(index='Value'))
        except:
            pass


    plot_choice = st.selectbox('Select type of plot',['','Box Plot','Correlation Plot','Pie Plot', 'Scatter Plot','Bar Plot'])

    if plot_choice == 'Box Plot':
        column_to_plot = st.selectbox("Select 1 Column", df.columns)
        fig = px.box(df,y=column_to_plot)
        st.plotly_chart(fig)

    if plot_choice =='Correlation Plot':
        plt.figure(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        st.pyplot(plt)

    if plot_choice =='Pie Plot':
        column_to_plot = st.selectbox("Select 1 Column", df.columns)
        value_counts = df[column_to_plot].value_counts()
        fig, ax = plt.subplots()
        ax.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis('equal')
        st.write(fig)

    if plot_choice =='Scatter Plot':
        try :
            selected_columns = st.multiselect('Select two columns',df.columns)
            first_column = selected_columns[0]
            second_column = selected_columns[1]
            fig = px.scatter(df, x=first_column, y=second_column)
            fig.update_layout(title="Scatter Plot", xaxis_title=first_column, yaxis_title=second_column)
            st.plotly_chart(fig)
        except:
            pass

    if plot_choice == 'Bar Plot':
        try :
            selected_columns = st.multiselect('Select columns', df.columns)
            first_column = selected_columns[0]
            second_column = selected_columns[1]
            fig = px.bar(df, x=first_column, y=second_column, title='Bar Plot')
            st.plotly_chart(fig)
        except :
            pass
    
if choice == "Profiling Report":
    df = pd.read_csv('dataset.csv', index_col=None)    
    profile = ProfileReport(df, title="New Data for profiling")
    st.subheader("Data Used for Report")
    st.write(df)
    st.subheader("Profiling Report")
    st_profile_report(profile)

    if st.button("Download Profiling Report"):
        report_filename = "profiling_report.html"
        profile.to_file(report_filename) 

        st.download_button(
            label="Download Profiling Report",
            data=report_filename,
            file_name="profiling_report.html",
            key="profiling_report_button",
        )

   
if choice == "Data Preparing" :
    df = pd.read_csv('dataset.csv', index_col=None) 

    st.title('Preparing the data before machine learning modeling')

    want_to_drop = st.selectbox('Do you want to drop any columns ?',['','Yes','No'])
    
    if want_to_drop == 'No':
        st.warning('It is recommended to drop columns such as name, customer ID, etc.')

    if want_to_drop == 'Yes':
        columns_to_drop = st.multiselect('Select columns to drop', df.columns)
        if columns_to_drop  :
            df = df.drop(columns_to_drop, axis=1)
            st.success('Columns dropped successfully.')
            st.dataframe(df)
            df.to_csv('dataset.csv', index=None)
            st.success('The new DataFrame has been successfully saved to "dataset.csv"')

    encoder_option = st.selectbox('Do you want to encode your data ?',['','Yes','No'])
    
    if encoder_option == 'No' :
        st.write('OK, Please processed to next step')
    
    if encoder_option == 'Yes' :
        encoder_columns = st.multiselect('Please pick the columns you want to encode',df.columns)
        encoder_type = st.selectbox('Please pick the type of encoder you want to use', ['','Label Encoder','One Hot Encoder'])
       
        if encoder_type == 'Label Encoder' :
            encoder = LabelEncoder()
            df[encoder_columns] = df[encoder_columns].apply(encoder.fit_transform)
            st.success('Columns encoded successfully.')
            st.dataframe(df)
            df.to_csv('dataset.csv', index=None)
            st.success('The new DataFrame has been successfully saved to "dataset.csv"')

        if encoder_type == 'One Hot Encoder':
            df = pd.get_dummies(df, columns=encoder_columns, prefix=encoder_columns,drop_first=True)
            st.success('Columns encoded successfully.')
            st.dataframe(df)
            df.to_csv('dataset.csv', index=None)
            st.success('The new DataFrame has been successfully saved to "dataset.csv"')


    fill_option = st.selectbox('Is there any missing data you want to fill ?', ['', 'Yes', 'No'])

    if fill_option == 'No':
        st.write('OK, Please processed to next step')

    if fill_option == 'Yes':
        encoder_columns = st.multiselect('Please pick the columns you want to fill', df.columns)
        encoder_type = st.selectbox('Please pick the type of filling you want to use', ['','Mean','Median','Most frequent'])

        try:
            if encoder_type == 'Mean' :

                imputer = SimpleImputer(strategy='mean')
                df[encoder_columns] = np.round(imputer.fit_transform(df[encoder_columns]),1)
                st.success('Selected columns filled successfully')
                st.dataframe(df)
                df.to_csv('dataset.csv', index=None)
                st.success('The new DataFrame has been successfully saved to "dataset.csv"')

            if encoder_type == 'Median' :

                imputer = SimpleImputer(strategy='median')
                df[encoder_columns] = np.round(imputer.fit_transform(df[encoder_columns]),1)
                st.success('Selected columns filled successfully')
                st.dataframe(df)
                df.to_csv('dataset.csv', index=None)
                st.success('The new DataFrame has been successfully saved to "dataset.csv"')

            if encoder_type == 'Most frequent' :

                imputer = SimpleImputer(strategy='most_frequent')
                df[encoder_columns] = np.round(imputer.fit_transform(df[encoder_columns]),1)
                st.success('Selected columns filled successfully')
                st.dataframe(df)
                df.to_csv('dataset.csv', index=None)
                st.success('The new DataFrame has been successfully saved to "dataset.csv"')
        except :
            pass

    scaling_option = st.selectbox('Do you want to scale your data?', ['', 'Yes', 'No'])
    if scaling_option == 'No':
        st.write('OK, Please proceed to the next step')
        st.dataframe(df)
    
    if scaling_option == 'Yes':
        scaler = MinMaxScaler()
        numeric_columns = df.select_dtypes(include='number').columns
        df_scaled = scaler.fit_transform(df[numeric_columns])
        df_scaled = pd.DataFrame(df_scaled, columns=numeric_columns)
        
        for column in df_scaled.columns:
            df_scaled[column] = pd.to_numeric(df_scaled[column], errors='coerce')

        st.success('The Data Frame has been successfully scaled')
        st.dataframe(df_scaled)

        df_scaled.to_csv('dataset.csv', index=None)
        st.success('The scaled DataFrame has been successfully saved to "dataset.csv"')
        
    
if choice == "Modelling":

    st.title('It is time for Machine Learning modeling')
    df = pd.read_csv('dataset.csv', index_col=None)

    target_choices = [''] + df.columns.tolist()

    try :
        target = st.selectbox('Choose your target variable', target_choices)
        X = df.drop(columns=target)
        y = df[target]
        st.write('Your Features are', X)
        st.write('Your Target is', y)

        test_size = st.select_slider('Pick the test size you want', range(1, 100, 1))
        st.warning('It is recommended to pick a number between 10 and 30 ')
        test_size_fraction = test_size / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_fraction, random_state=42)
        st.write('Shape of training data is :', X_train.shape)
        st.write('Shape of testing data is :', X_test.shape)

    except :
        pass

    task_type = st.selectbox('Choose type of task you want to apply', ['','Classification', 'Regression'])
    

    if task_type == 'Classification':

                algo_type = st.selectbox('Please choose which type of algorithm you want to use',
                                         ['','Logistic Regression','Decision Trees','Random Forest','SVC','KNN'])

                if algo_type == 'Logistic Regression' :

                    from sklearn.linear_model import LogisticRegression

                    clf = LogisticRegression(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)


                if algo_type == 'Decision Trees' :

                    from sklearn.tree import DecisionTreeClassifier

                    clf = DecisionTreeClassifier(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                if algo_type == 'Random Forest' :

                    from sklearn.ensemble import RandomForestClassifier

                    clf = RandomForestClassifier(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                if algo_type == 'SVC' :

                    from sklearn.svm import SVC

                    clf = SVC(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                if algo_type == 'KNN' :

                    from sklearn.neighbors import KNeighborsClassifier

                    clf = KNeighborsClassifier()
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                evaluation_type = st.selectbox('Choose type of evaluation metrics ',['','Accuracy','Confusion Matrix',
                                                                                 'Precision, Recall, and F1-score'])
                
                if evaluation_type == 'Accuracy' :
                    from sklearn.metrics import accuracy_score
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write("Accuracy:", accuracy)
                                
                if evaluation_type == 'Confusion Matrix' :
                    from sklearn.metrics import confusion_matrix
                    cm = confusion_matrix(y_test, y_pred)
                    st.write("Confusion Matrix:")
                    st.dataframe(cm)
                    
                if evaluation_type == 'Precision, Recall, and F1-score' :
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    
                    metrics_dict = {"Metric": ["Precision", "Recall", "F1-Score"], "Value": [precision, recall, f1]}
                    metrics_df = pd.DataFrame(metrics_dict)
                    st.dataframe(metrics_df)
                
                try :
                    model_filename = "clf.pkl"
                    with open(model_filename, "wb") as model_file:
                        pickle.dump(clf, model_file)
                        st.download_button('Download the model', open(model_filename, 'rb').read(), 'clf.pkl')
                
                except :
                    pass


    if task_type == 'Regression':


            algo_type = st.selectbox('Please choose which type of algorithm you want to use',
                                     ['','Linear Regression','Ridge','SVR','Random Forest'])

            if algo_type == 'Linear Regression' :

                from sklearn.linear_model import LinearRegression

                rg = LinearRegression()
                rg.fit(X_train, y_train)
                y_pred = rg.predict(X_test)


            if algo_type == 'Ridge' :

                from sklearn.linear_model import Ridge

                rg = Ridge()
                rg.fit(X_train, y_train)
                y_pred = rg.predict(X_test)


            if algo_type == 'SVR' :

                from sklearn.svm import SVR

                rg = SVR()
                rg.fit(X_train, y_train)
                y_pred = rg.predict(X_test)

            if algo_type == 'Random Forest' :

                from sklearn.ensemble import RandomForestRegressor

                rg = RandomForestRegressor()
                rg.fit(X_train, y_train)
                y_pred = rg.predict(X_test)

            evaluation_type = st.selectbox('Choose type of evaluation metrics ',['','MAE','MSE','r2 score'])

            if evaluation_type == 'MAE' :

                from sklearn.metrics import mean_absolute_error

                MAE = mean_absolute_error(y_test, y_pred)
                st.write("Mean absolute error:", MAE)

            if evaluation_type == 'MSE' :

                from sklearn.metrics import mean_squared_error

                MSE = mean_squared_error(y_test, y_pred)
                st.write("Mean squared error:", MSE)

            if evaluation_type == 'r2 score' :

                from sklearn.metrics import r2_score

                r2 = r2_score(y_test, y_pred)
                st.write("r2 score:", r2)


            try :

                model_filename = "rg.pkl"
                with open(model_filename, "wb") as model_file:
                    pickle.dump(rg, model_file)

                st.download_button('Download the model', open(model_filename, 'rb').read(), 'rg.pkl')

            except :
                pass             
                                                                          


if choice == "Analysis":

    st.title('Comparative Analysis')
    df = pd.read_csv('dataset1.csv', index_col=None)

    target_choices = [''] + df.columns.tolist()

    try :
        target = st.selectbox('Choose your target variable', target_choices)
        X = df.drop(columns=target)
        y = df[target]
        st.write('Your Features are', X)
        st.write('Your Target is', y)

        test_size = st.select_slider('Pick the test size you want', range(1, 100, 1))
        st.warning('It is recommended to pick a number between 10 and 30 ')
        test_size_fraction = test_size / 100.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_fraction, random_state=42)
        st.write('Shape of training data is :', X_train.shape)
        st.write('Shape of testing data is :', X_test.shape)

    except :
        pass

    task_type = st.selectbox('Choose type of task you want to apply', ['','Classification', 'Regression'])
    

    if task_type == 'Classification':

            from pycaret.classification import *

            if st.button('Run Modelling'):

                setup(df, target=target, verbose=False)
                setup_df = pull()
                st.info("This is the ML experiment settings")
                st.dataframe(setup_df)
                best_model = compare_models()
                compare_df = pull()
                st.info("This is your ML model")
                st.dataframe(compare_df)
                save_model(best_model, 'best_model')

                with open('best_model.pkl', 'rb') as model_file:
                    st.download_button('Download the model', model_file, 'best_model.pkl')


    if task_type == 'Regression':

            from pycaret.regression import *

            if st.button('Run Modelling'):

                setup(df, target=target, verbose=False)
                setup_df = pull()
                st.info("This is the ML experiment settings")
                st.dataframe(setup_df)
                best_model = compare_models()
                compare_df = pull()
                st.info("This is your ML model")
                st.dataframe(compare_df)
                save_model(best_model, 'best_model')

                with open('best_model.pkl', 'rb') as model_file:
                    st.download_button('Download the model', model_file, 'best_model.pkl')

    st.image("3.jpeg")

if choice == "Download":
    st.title("Download files here.")
    st.image("4.jpeg")
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
