# %% [markdown]
# ## Name :Shreyash Jaiswal

# %% [markdown]
# # Model Building task

# %% [markdown]
# * #### Logistic Regression model

# %% [markdown]
# * Importing libraries

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:24.243422Z","iopub.execute_input":"2023-03-24T15:04:24.243829Z","iopub.status.idle":"2023-03-24T15:04:24.248940Z","shell.execute_reply.started":"2023-03-24T15:04:24.243793Z","shell.execute_reply":"2023-03-24T15:04:24.247745Z"}}
# import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# %% [markdown]
# * Loading data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:24.250454Z","iopub.execute_input":"2023-03-24T15:04:24.250794Z","iopub.status.idle":"2023-03-24T15:04:24.724828Z","shell.execute_reply.started":"2023-03-24T15:04:24.250761Z","shell.execute_reply":"2023-03-24T15:04:24.723134Z"}}
# importing data
df = pd.read_csv('/kaggle/input/vallari-feature-engg/data_feature_eng_final.csv')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:24.726376Z","iopub.execute_input":"2023-03-24T15:04:24.726750Z","iopub.status.idle":"2023-03-24T15:04:24.745528Z","shell.execute_reply.started":"2023-03-24T15:04:24.726712Z","shell.execute_reply":"2023-03-24T15:04:24.744031Z"}}
# head of the data
print(df.head())

# %% [markdown]
# * Now we will split our data into training and testing sets and make the data ready for the machine learning algorithm.

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:24.748420Z","iopub.execute_input":"2023-03-24T15:04:24.748894Z","iopub.status.idle":"2023-03-24T15:04:24.758640Z","shell.execute_reply.started":"2023-03-24T15:04:24.748836Z","shell.execute_reply":"2023-03-24T15:04:24.756617Z"}}
df.columns

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:24.760241Z","iopub.execute_input":"2023-03-24T15:04:24.760944Z","iopub.status.idle":"2023-03-24T15:04:24.773858Z","shell.execute_reply.started":"2023-03-24T15:04:24.760903Z","shell.execute_reply":"2023-03-24T15:04:24.772428Z"}}
df.Age

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:24.775286Z","iopub.execute_input":"2023-03-24T15:04:24.776605Z","iopub.status.idle":"2023-03-24T15:04:24.791219Z","shell.execute_reply.started":"2023-03-24T15:04:24.776441Z","shell.execute_reply":"2023-03-24T15:04:24.789056Z"}}
df.Interest

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:24.793680Z","iopub.execute_input":"2023-03-24T15:04:24.794483Z","iopub.status.idle":"2023-03-24T15:04:24.807401Z","shell.execute_reply.started":"2023-03-24T15:04:24.794410Z","shell.execute_reply":"2023-03-24T15:04:24.805978Z"}}
df.LoanDuration

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:24.810533Z","iopub.execute_input":"2023-03-24T15:04:24.811023Z","iopub.status.idle":"2023-03-24T15:04:24.823249Z","shell.execute_reply.started":"2023-03-24T15:04:24.810984Z","shell.execute_reply":"2023-03-24T15:04:24.821427Z"}}
df.NoOfPreviousLoansBeforeLoan

# %% [code] {"execution":{"iopub.status.busy":"2023-03-24T15:04:24.828365Z","iopub.execute_input":"2023-03-24T15:04:24.829123Z","iopub.status.idle":"2023-03-24T15:04:24.834506Z","shell.execute_reply.started":"2023-03-24T15:04:24.829070Z","shell.execute_reply":"2023-03-24T15:04:24.833136Z"}}
features= ['Age','Interest','LoanDuration','NoOfPreviousLoansBeforeLoan'] 

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:24.836211Z","iopub.execute_input":"2023-03-24T15:04:24.836711Z","iopub.status.idle":"2023-03-24T15:04:24.852877Z","shell.execute_reply.started":"2023-03-24T15:04:24.836662Z","shell.execute_reply":"2023-03-24T15:04:24.851240Z"}}
#split dataset into independent and dependent feature
y = df.Defaulted
X = df[features]
#X = df.drop(columns = ['Defaulted'])

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:24.854826Z","iopub.execute_input":"2023-03-24T15:04:24.855576Z","iopub.status.idle":"2023-03-24T15:04:24.874983Z","shell.execute_reply.started":"2023-03-24T15:04:24.855525Z","shell.execute_reply":"2023-03-24T15:04:24.873917Z"}}
X

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:24.876673Z","iopub.execute_input":"2023-03-24T15:04:24.877015Z","iopub.status.idle":"2023-03-24T15:04:24.891639Z","shell.execute_reply.started":"2023-03-24T15:04:24.876983Z","shell.execute_reply":"2023-03-24T15:04:24.890555Z"}}
y

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:24.893143Z","iopub.execute_input":"2023-03-24T15:04:24.893561Z","iopub.status.idle":"2023-03-24T15:04:24.914736Z","shell.execute_reply.started":"2023-03-24T15:04:24.893523Z","shell.execute_reply":"2023-03-24T15:04:24.913598Z"}}
# Let's use Train Test Split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,
                                                    train_size = 0.8)

# %% [markdown]
# * Logistic Regression model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:24.916160Z","iopub.execute_input":"2023-03-24T15:04:24.916539Z","iopub.status.idle":"2023-03-24T15:04:24.923766Z","shell.execute_reply.started":"2023-03-24T15:04:24.916504Z","shell.execute_reply":"2023-03-24T15:04:24.922246Z"}}
# Import Logistic Regression model
from sklearn.linear_model import LogisticRegression

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:24.926404Z","iopub.execute_input":"2023-03-24T15:04:24.926998Z","iopub.status.idle":"2023-03-24T15:04:24.938358Z","shell.execute_reply.started":"2023-03-24T15:04:24.926944Z","shell.execute_reply":"2023-03-24T15:04:24.936434Z"}}
# Let's make LogisticRegression object
LR = LogisticRegression()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:24.939763Z","iopub.execute_input":"2023-03-24T15:04:24.940158Z","iopub.status.idle":"2023-03-24T15:04:25.502887Z","shell.execute_reply.started":"2023-03-24T15:04:24.940121Z","shell.execute_reply":"2023-03-24T15:04:25.501105Z"}}
# Let's fit the model on our data
LR.fit(X_train, y_train)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:25.511532Z","iopub.execute_input":"2023-03-24T15:04:25.517200Z","iopub.status.idle":"2023-03-24T15:04:25.532869Z","shell.execute_reply.started":"2023-03-24T15:04:25.517101Z","shell.execute_reply":"2023-03-24T15:04:25.531058Z"}}
# Let's make predictions
y_predict = LR.predict(X_test)

# %% [markdown]
# * Model Evaluation

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:25.540742Z","iopub.execute_input":"2023-03-24T15:04:25.542616Z","iopub.status.idle":"2023-03-24T15:04:25.557849Z","shell.execute_reply.started":"2023-03-24T15:04:25.542534Z","shell.execute_reply":"2023-03-24T15:04:25.556111Z"}}
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:25.559936Z","iopub.execute_input":"2023-03-24T15:04:25.569777Z","iopub.status.idle":"2023-03-24T15:04:25.593943Z","shell.execute_reply.started":"2023-03-24T15:04:25.569684Z","shell.execute_reply":"2023-03-24T15:04:25.592238Z"}}
# accuracy_score of LogisticResgression model
accuracy_score(y_test, y_predict)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:25.603806Z","iopub.execute_input":"2023-03-24T15:04:25.609086Z","iopub.status.idle":"2023-03-24T15:04:25.643026Z","shell.execute_reply.started":"2023-03-24T15:04:25.608990Z","shell.execute_reply":"2023-03-24T15:04:25.641088Z"}}
roc_auc_score(y_test, y_predict)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:25.650918Z","iopub.execute_input":"2023-03-24T15:04:25.651302Z","iopub.status.idle":"2023-03-24T15:04:25.662880Z","shell.execute_reply.started":"2023-03-24T15:04:25.651267Z","shell.execute_reply":"2023-03-24T15:04:25.661566Z"}}
# confusion matrix
confusion_matrix(y_test, y_predict)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:25.664754Z","iopub.execute_input":"2023-03-24T15:04:25.665224Z","iopub.status.idle":"2023-03-24T15:04:25.963814Z","shell.execute_reply.started":"2023-03-24T15:04:25.665189Z","shell.execute_reply":"2023-03-24T15:04:25.962554Z"}}
# plot confusion matrix 
plot_confusion_matrix(LR, X_test, y_test);

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-03-24T15:04:25.965931Z","iopub.execute_input":"2023-03-24T15:04:25.966740Z","iopub.status.idle":"2023-03-24T15:04:25.977086Z","shell.execute_reply.started":"2023-03-24T15:04:25.966690Z","shell.execute_reply":"2023-03-24T15:04:25.975511Z"}}
#import lib used saving for model
from joblib import parallel, delayed
import joblib

#saving the model as a pickle in a file
joblib.dump(LR,'LR model.pkl')
#load with
#model = joblib.load('mymodel.pkl')