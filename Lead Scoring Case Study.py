#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required Libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.model_selection import learning_curve
import warnings
from sklearn.model_selection import cross_validate

warnings.filterwarnings("ignore")


# In[2]:


plt.style.use("ggplot")


# In[3]:


get_ipython().system('pip install nb-black')


# In[4]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# In[5]:


# The "get_vif_score" function takes a pandas DataFrame of training data as input and returns a DataFrame with the Variance Inflation Factor (VIF) scores for each column.

# Input Parameters:

# X_train: A pandas DataFrame containing the training data for which the VIF scores need to be calculated.

# Output Parameters:

# vif_df: A pandas DataFrame with two columns: "features", which lists the columns of the input data, and "VIF", which lists the corresponding VIF scores.


def get_vif_score(X_train):
    vif_df = pd.DataFrame()
    vif_df["features"] = X_train.columns
    vif_df["VIF"] = [vif(X_train.values, i) for i in range(X_train.shape[1])]
    vif_df["VIF"] = round(vif_df["VIF"], 2)
    return vif_df


# The "get_rfe_scores" function takes in a number of features to be selected, a training dataset, and a target variable, and returns a list of the top features selected using Recursive Feature Elimination (RFE) with Linear Regression.

# Input Parameters:

# no_of_features_to_be_selected: An integer value indicating the number of top features to be selected using RFE.
# X_train: A pandas DataFrame containing the training data.
# y_train: A pandas Series containing the target variable for the training data.
#
# Output Parameters:

# finalcols: A list of the column names corresponding to the top features selected by RFE with Linear Regression.


def get_rfe_scores(no_of_features_to_be_selected, X_train, y_train):
    n_features_to_select = no_of_features_to_be_selected
    lm = LogisticRegression()
    rfe = RFE(estimator=lm, n_features_to_select=n_features_to_select)
    rfe = rfe.fit(X_train, y_train)
    finalcols = []

    for x, y, z in sorted(list(zip(X_train.columns, rfe.support_, rfe.ranking_))):
        if y == True:
            finalcols.append(x)

        # print(x, y, z)
    return list(finalcols)


# In[6]:


# The "build_model_glm" function takes in a pandas DataFrame of input variables (X), a pandas Series of the target variable (y), and a list of features to be used in the model, and returns a trained Ordinary Least Squares (OLS) model using statsmodels.

# Input parameters:
#     X: a pandas DataFrame containing the feature variables
#     y: a pandas Series containing the response variable
#     model_features: a list of column names to use as features in the model
# Output:
#     A tuple containing:
#         A summary of the model fit using the summary method of the fitted GLM model object
#         The fitted GLM model object itself
#         A pandas DataFrame with the constant column added to X[model_features] called X_train_sm


def build_model_glm(X, y, model_features):
    X_train_sm = sm.add_constant(X[model_features])
    lr = sm.GLM(
        y,
        X_train_sm,
        family=sm.families.Binomial(),
    )

    #     X_train_sm = sm.add_constant(X_train[sm5cols])
    #     logm1 = sm.GLM(y_train, (X_train_sm), family=sm.families.Binomial())
    #     res = logm1.fit()
    #     res.summary()

    return lr.fit().summary(), lr.fit(), X_train_sm


# In[7]:


# This function plots the distribution of error terms between actual and predicted values.

# Input parameters: actual (act) and predicted (pred) values of a model

# Output: a plot of the distribution of error terms between actual and predicted values.


def plot_res_dist(act, pred):
    plt.figure(figsize=(20, 10))
    sns.distplot(act - pred)
    plt.title("Error Terms", fontsize=20)
    plt.xlabel("Errors")


# The "plot_scatter" function takes in a pandas Series of X values, a pandas Series of y values, labels for the X and y axes, and a title for the plot, and generates a scatter plot using Matplotlib.

# Input Parameters:

# X: A pandas Series of X values to be plotted.
# y: A pandas Series of y values to be plotted.
# Xlabel: A string representing the label for the X axis.
# Ylabel: A string representing the label for the y axis.
# Title: A string representing the title for the plot.

# Output Parameters:

# None. The function generates a plot using Matplotlib.


def plot_scatter(X, y, Xlabel, Ylabel, Title):
    plt.figure(figsize=(20, 10))
    plt.scatter(X, y, alpha=0.5, s=60)
    plt.ylabel(f"{Xlabel}")
    plt.xlabel(f"{Ylabel}")
    plt.title(f"{Title}", fontsize=20)
    plt.show()


# In[8]:


# This function creates bar plots for all categorical columns in a DataFrame against the binary target column 'Converted'.

# Input:
# A pandas DataFrame with categorical columns and a binary target column named 'Converted'.

# Output:
# A series of bar plots, one for each categorical column, showing the count of each unique value in the column and the number of converted and non-converted instances for each value.


def plot_bar_categorical_with_convertedcol(df):
    string_cols = [
        i
        for i in df.columns[df.dtypes == "object"]
        if i not in ("Prospect ID", "Lead Number")
    ]
    for i in string_cols:
        print(i)
        plt.figure(figsize=(10, 5))
        s1 = sns.countplot(df[i], hue=df.Converted)
        s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
        plt.show()


# In[9]:


# Summary: This function generates a learning curve plot to visualize the accuracy of a machine learning model at different sizes of training data. It uses the learning_curve function from scikit-learn to calculate the mean training and validation scores for different training data sizes, and then plots the results using matplotlib.

# Input parameters:

#     model: The machine learning model to use for training.
#     X_train: The feature matrix of the training data.
#     y_train: The target vector of the training data.

# Output parameters: None. The function generates a learning curve plot using matplotlib.


def plot_learning_curve(model, X_train, y_train):

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=10,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    print(train_mean, train_std, test_mean, test_std)

    plt.figure(figsize=(20, 10))

    plt.plot(
        train_sizes,
        train_mean,
        color="blue",
        marker="o",
        markersize=5,
        label="Training Accuracy",
    )
    plt.fill_between(
        train_sizes,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="blue",
    )
    plt.plot(
        train_sizes,
        test_mean,
        color="green",
        marker="+",
        markersize=5,
        linestyle="--",
        label="Validation Accuracy",
    )
    plt.fill_between(
        train_sizes,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color="green",
    )
    plt.title("Learning Curve", fontsize=20)
    plt.xlabel("Training Data Size")
    plt.ylabel("Model accuracy")
    plt.grid()
    plt.legend(loc="lower right")
    plt.show()


# In[10]:


# Summary:

# Function name: model_metrics
# Takes a confusion matrix as input and returns four evaluation metrics for a classification model: precision, accuracy, recall, and F1 score.

# Input parameters:

#     Actual: Represents the actual labels of the data.
#     Predicted: Represents the predicted labels generated by the classification model.

# Output parameters:

#     A tuple that contains the following four metrics:
#     Accuracy: The proportion of true results (both true positives and true negatives) among the total number of cases.
#     Precision: The proportion of true positives out of the total predicted positives.
#     Recall: The proportion of true positives out of the total actual positives.
#     F1 score: The harmonic mean of precision and recall.


def model_metrics(Actual, Predicted):
    Accuracy = metrics.accuracy_score(Actual, Predicted)

    confusion = metrics.confusion_matrix(Actual, Predicted)

    TP = confusion[1, 1]  # true positive
    TN = confusion[0, 0]  # true negatives
    FP = confusion[0, 1]  # false positives
    FN = confusion[1, 0]  # false negatives

    return (
        Accuracy,
        (TP / (TP + FN)),
        (TN / (TN + FP)),
        (TP / (TP + FP)),
        (TP / (TP + FN)),
    )


# In[11]:


# Summary:

# Function name: cross_validation
# Performs cross-validation for a given model and returns the training and validation metrics for accuracy, precision, recall, and F1 score.

# Input parameters:

#     model: The classification model to be evaluated using cross-validation.
#     _X: The input features of the dataset.
#     _y: The target labels of the dataset.
#     _cv (optional): The number of folds to be used in cross-validation. The default value is 5.

# Output parameters:

#     A dictionary that contains the following metrics:
#     Training Accuracy scores: The accuracy scores for each fold of the training data.
#     Mean Training Accuracy: The mean accuracy score of the training data.
#     Training Precision scores: The precision scores for each fold of the training data.
#     Mean Training Precision: The mean precision score of the training data.
#     Training Recall scores: The recall scores for each fold of the training data.
#     Mean Training Recall: The mean recall score of the training data.
#     Training F1 scores: The F1 scores for each fold of the training data.
#     Mean Training F1 Score: The mean F1 score of the training data.
#     Validation Accuracy scores: The accuracy scores for each fold of the validation data.
#     Mean Validation Accuracy: The mean accuracy score of the validation data.
#     Validation Precision scores: The precision scores for each fold of the validation data.
#     Mean Validation Precision: The mean precision score of the validation data.
#     Validation Recall scores: The recall scores for each fold of the validation data.
#     Mean Validation Recall: The mean recall score of the validation data.
#     Validation F1 scores: The F1 scores for each fold of the validation data.
#     Mean Validation F1 Score: The mean F1 score of the validation data.


def cross_validation(model, _X, _y, _cv=5):

    _scoring = ["accuracy", "precision", "recall", "f1"]
    results = cross_validate(
        estimator=model, X=_X, y=_y, cv=_cv, scoring=_scoring, return_train_score=True
    )

    return {
        "Training Accuracy scores": results["train_accuracy"],
        "Mean Training Accuracy": results["train_accuracy"].mean() * 100,
        "Training Precision scores": results["train_precision"],
        "Mean Training Precision": results["train_precision"].mean(),
        "Training Recall scores": results["train_recall"],
        "Mean Training Recall": results["train_recall"].mean(),
        "Training F1 scores": results["train_f1"],
        "Mean Training F1 Score": results["train_f1"].mean(),
        "Validation Accuracy scores": results["test_accuracy"],
        "Mean Validation Accuracy": results["test_accuracy"].mean() * 100,
        "Validation Precision scores": results["test_precision"],
        "Mean Validation Precision": results["test_precision"].mean(),
        "Validation Recall scores": results["test_recall"],
        "Mean Validation Recall": results["test_recall"].mean(),
        "Validation F1 scores": results["test_f1"],
        "Mean Validation F1 Score": results["test_f1"].mean(),
    }


# In[12]:


# Summary:

# Function name: plot_result
# Plots a bar chart to show the comparison between training and validation data for a given metric.

# Input parameters:

#     x_label: The label for the x-axis of the plot.
#     y_label: The label for the y-axis of the plot.
#     plot_title: The title of the plot.
#     train_data: A list of the training data for the given metric, with one value for each fold.
#     val_data: A list of the validation data for the given metric, with one value for each fold.

# Output parameters:

#     The function does not have any return value, but it plots a bar chart that compares the training and validation data for a given metric. The chart shows the values for each fold of the data, and the mean value for both training and validation data.


def plot_result(x_label, y_label, plot_title, train_data, val_data):
    # Set size of plot
    plt.figure(figsize=(12, 6))
    labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
    X_axis = np.arange(len(labels))
    ax = plt.gca()
    plt.ylim(0.40000, 1)
    plt.bar(X_axis - 0.2, train_data, 0.4, color="blue", label="Training")
    plt.bar(X_axis + 0.2, val_data, 0.4, color="red", label="Validation")
    plt.title(plot_title, fontsize=30)
    plt.xticks(X_axis, labels)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()


# In[13]:


# Summary:
# Function name: create_boxplot_numberical_with_converted
# Creates boxplots to compare the distribution of numerical variables between different groups (e.g., between converted and non-converted customers).

# Input parameters:
#     df: A Pandas DataFrame containing the data to be plotted. The DataFrame should include at least one numerical column and one column representing the groups to be compared (e.g., "Converted" column with binary values).

# Output parameters:
#     The function does not have any return value, but it creates a boxplot for each numerical column in the input DataFrame. Each boxplot shows the distribution of the numerical variable for the different groups defined in the "Converted" column. The boxplot allows us to compare the median, quartiles, and outliers of the numerical variable between the different groups.


def create_boxplot_numberical_with_converted(df):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    num_cols2 = [i for i in num_cols if i not in ["Lead Number", "Converted"]]

    for i in num_cols2:
        sns.boxplot(y=i, x="Converted", data=df)
        plt.show()


# In[14]:


# Function to plot the receiver operating characteristic (ROC) curve for a binary classification model.

# Input parameters
#     actual: array-like of shape (n_samples,)
#     True binary labels of the test data.

#     probs: array-like of shape (n_samples,)
#     Estimated probabilities or decision function of the classifier.

# Output parameters
#     The function displays the ROC curve plot with area under the curve (AUC) score.


def draw_roc(actual, probs):
    fpr, tpr, thresholds = metrics.roc_curve(actual, probs, drop_intermediate=False)
    auc_score = metrics.roc_auc_score(actual, probs)
    plt.figure(figsize=(20, 15))
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % auc_score)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate or [1 - True Negative Rate]")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()

    return None


# ## A. Data Sourcing 

# In[15]:


df = pd.read_csv(
    r"C:\Users\sharmaanuj\Downloads\Lead+Scoring+Case+Study\Lead Scoring Assignment\Leads.csv"
)
df.head(10)


# In[16]:


df.shape


# In[17]:


df.describe()


# ## B. Explortatory Data Analysis

# #### 1. Missing Value Imputation and Data Cleaning

# In[18]:


df.isnull().sum().sort_values(ascending=False) / df.shape[0] * 100


# In[19]:


## Replacing all the values which has Select value would be replaced by Null.

df = df.replace("Select", np.nan)


# In[20]:


df.isnull().sum().sort_values(ascending=False) / df.shape[0] * 100


# In[21]:


# Dropping cols with more than 40% missing values

cols = df.columns

for i in cols:
    if (100 * (df[i].isnull().sum() / len(df.index))) >= 40:
        df.drop(i, 1, inplace=True)


# In[22]:


df.isnull().sum().sort_values(ascending=False) / df.shape[0] * 100


# In[23]:


df["Tags"].value_counts(dropna=False)


# In[24]:


df["Tags"] = df["Tags"].replace(np.nan, "Others")


# In[25]:


## Given that there are numerous values in the tag column, we have performed a classification process to group them into clusters that contain tags of similar types.

df["Tags"] = df["Tags"].replace(
    [        "Ringing",        "switched off",        "invalid number",        "wrong number given",        "number not provided",    ],
    "Not Reachable"
).replace(
    [        "Not Reachable",        "Closed by Horizzon",        "Not doing further education",        "Graduation in progress",        "Diploma holder (Not Eligible)",        "Lost to Others",        "University not recognized",        "Recognition issue (DEC approval)",    ],
    "Not interested or unable to continue"
).replace(
    [        "Will revert after reading the email",        "Interested in other courses",        "Interested in full time MBA",        "Interested  in full time MBA",        "Still Thinking",        "Want to take admission but has financial problems",        "In confusion whether part time or DLP",        "Interested in Next batch",        "Shall take in the next coming month",    ],
    "Interested or considering"
).replace(
    [        "Already a student",    ],
    "Enrolled or already a student"
).replace(
    ["Lost to EINS", "in touch with EINS"],
    "Contacted or in touch with EINS"
).replace(
    [        "Busy",    ],
    "Busy or unavailable"
).replace(
    ["opp hangup", "Lateral student"],
    "Other information provided"
)


# In[26]:


df["Tags"].value_counts(dropna=False).plot.bar()
plt.show()


# In[27]:


df["Last Activity"].value_counts(dropna=False)


# In[28]:


df["Last Activity"] = df["Last Activity"].replace(np.nan, "Unknown")


# In[29]:


df["Last Activity"] = (
    df["Last Activity"]
    .replace(
        [
            "Email Opened",
            "SMS Sent",
            "Resubscribed to emails",
            "Visited Booth in Tradeshow",
            "Email Received",
            "View in browser link Clicked",
            "Form Submitted on Website",
            "Olark Chat Conversation",
            "Page Visited on Website",
            "Converted to Lead",
            "Email Link Clicked",
        ],
        "Engaged with the brand or product",
    )
    .replace(
        ["Unknown", "Unreachable", "Email Bounced"], "Unreachable or unable to contact"
    )
    .replace(["Unsubscribed"], "No longer interested")
    .replace(["Had a Phone Conversation"], "Had a conversation")
    .replace(["Approached upfront"], "Approached directly")
    .replace(["Email Marked Spam"], "Other information provided")
)


# In[30]:


df["Last Activity"].value_counts(dropna=False)


# In[31]:


df["Last Activity"].value_counts(dropna=False).plot.bar()
plt.show()


# In[32]:


df["Lead Source"].value_counts(dropna=False)


# In[33]:


LeadSource = df["Lead Source"].mode()[0]


# In[34]:


df["Lead Source"] = df["Lead Source"].replace(np.nan, LeadSource)


# In[35]:


df['Lead Source'] = df['Lead Source'].replace(
    ['Google', 'Direct Traffic', 'Organic Search', 'Reference', 'Referral Sites', 'bing', 'Click2call', 'Pay per Click Ads', 'welearnblog_Home', 'WeLearn', 'blog', 'NC_EDM'], 
    'Traffic Source'
).replace(
    ['Olark Chat', 'Live Chat'], 
    'Chat Platform'
).replace(
    ['Facebook', 'Social Media', 'youtubechannel'], 
    'Social Media'
).replace(
    'Welingak Website', 
    'Website'
).replace(
    ['Press_Release', 'testone'], 
    'Other'
)


# In[36]:


df["Lead Source"].value_counts(dropna=False).plot.bar()
plt.show()


# In[37]:


df["What matters most to you in choosing a course"].value_counts(dropna=False)


# In[38]:


df["What matters most to you in choosing a course"] = df[
    "What matters most to you in choosing a course"
].replace(np.nan, "Better Career Prospects")


# In[39]:


df["What matters most to you in choosing a course"].value_counts(dropna=False)


# In[40]:


df["What is your current occupation"].value_counts(dropna=False)


# In[41]:


df["What is your current occupation"] = df["What is your current occupation"].replace(
    np.nan, "Unemployed"
)


# In[42]:


df["Country"].value_counts(dropna=False)


# In[43]:


Country = df["Country"].mode()[0]


# In[44]:


df["Country"] = df["Country"].replace(np.nan, Country)


# In[45]:


df["Country"].value_counts(dropna=False)


# In[46]:


df["Specialization"].value_counts(dropna=False)


# In[47]:


df["Specialization"] = df["Specialization"].replace(np.nan, "Not Specified")


# In[48]:


mapping = {
    'Finance Management': 'Finance',
    'Banking, Investment And Insurance': 'Finance',
    'Human Resource Management': 'Human Resources',
    'Marketing Management': 'Marketing',
    'Media and Advertising': 'Marketing',
    'Operations Management': 'Operations',
    'Supply Chain Management': 'Operations',
    'Rural and Agribusiness': 'Operations',
    'IT Projects Management': 'Information Technology',
    'Business Administration': 'Business Administration',
    'Healthcare Management': 'Healthcare',
    'Hospitality Management': 'Hospitality',
    'Retail Management': 'Retail',
    'Not Specified': 'Not Specified',
    'Travel and Tourism': 'Travel and Tourism',
    'Media and Advertising': 'Media and Advertising',
    'International Business': 'International Business',
    'E-COMMERCE': 'E-commerce',
    'E-Business': 'E-business',
    'Services Excellence': 'Services'
}

df['Specialization'].replace(mapping, inplace=True)


# In[49]:


df["Specialization"].value_counts(dropna=False)


# In[50]:


df["Last Notable Activity"].value_counts()


# In[51]:


df['Last Notable Activity'] = df['Last Notable Activity'].replace({'Email Opened': 'Email Interactions',
                                                                   'Email Link Clicked': 'Email Interactions',
                                                                   'Email Bounced': 'Email Interactions',
                                                                   'Email Marked Spam': 'Email Interactions',
                                                                   'Email Received': 'Email Interactions',
                                                                   'Page Visited on Website': 'Website Interactions',
                                                                   'View in browser link Clicked': 'Website Interactions',
                                                                   'Form Submitted on Website': 'Website Interactions',
                                                                   'Olark Chat Conversation': 'Chat Interactions',
                                                                   'SMS Sent': 'Communication Interactions',
                                                                   'Had a Phone Conversation': 'Communication Interactions',
                                                                   'Approached upfront': 'Communication Interactions',
                                                                   'Unsubscribed': 'Subscription Interactions',
                                                                   'Resubscribed to emails': 'Subscription Interactions',
                                                                   'Unreachable': 'Unreachable Interactions'})


# In[52]:


df["Last Notable Activity"].value_counts()


# In[53]:


df["City"].value_counts(dropna=False)


# In[54]:


city = df["City"].mode()[0]


# In[55]:


df["City"] = df["City"].replace(np.nan, city)


# In[56]:


df["Lead Origin"].value_counts(dropna=False)


# In[57]:


fig, ax = plt.subplots(figsize=(10, 5))

# Define the labels and colors for the bars
labels_with_yesno_values = [
    "Do Not Email",
    "Do Not Call",
    "Search",
    "Magazine",
    "Newspaper Article",
    "X Education Forums",
    "Newspaper",
    "Digital Advertisement",
    "Through Recommendations",
    "Receive More Updates About Our Courses",
    "Update me on Supply Chain Content",
    "Get updates on DM Content",
    "I agree to pay the amount through cheque",
    "A free copy of Mastering The Interview",
]

for i in labels_with_yesno_values:
    df[i].value_counts().plot.pie()
    plt.show()


# In[58]:


df.head()


# In[59]:


df["Page Views Per Visit"].describe()


# In[60]:


df["Page Views Per Visit"].mean()


# In[61]:


df["Page Views Per Visit"].median()


# In[62]:


df["Page Views Per Visit"].quantile([0.2, 0.4, 0.5, 0.6, 0.8, 0.9])


# In[63]:


df["Page Views Per Visit"].plot.box()


# In[64]:


pagevisit = df["Page Views Per Visit"].median()

df["Page Views Per Visit"] = df["Page Views Per Visit"].replace(np.nan, pagevisit)


# In[65]:


df["TotalVisits"].mean()


# In[66]:


df["TotalVisits"].quantile([0.2, 0.4, 0.5, 0.6, 0.8, 0.9])


# In[67]:


df["TotalVisits"].plot.box()


# In[68]:


TotalVisits = df["TotalVisits"].median()

df["TotalVisits"] = df["TotalVisits"].replace(np.nan, TotalVisits)


# In[69]:


df.info()


# In[70]:


df.isnull().sum().sort_values(ascending=False) / df.shape[0] * 100


# In[71]:


df.shape


# #### 2. Explortatory Data Analysis

# #### Categorical Attributes Analysis

# In[72]:


plot_bar_categorical_with_convertedcol(df)
plt.show()


# In[73]:


df.head()


# ### Inference
# 
# 1.	Based on LeadSource column, it is observed that the majority of leads are sourced from chat platforms and traffic sources such as Google, Direct Traffic, Organic Search, Reference, Referral Sites, Bing, Click2call, Pay per Click Ads, welearnblog_Home, WeLearn, blog, and NC_EDM.
# 2.	Interaction Preference column suggests that about 90% of users prefer not to be contacted via phone or email regarding the product.
# 3.	Last Activity column analysis indicates that more than 80% of users who have been engaged through activities such as "Email Opened", "SMS Sent", "Resubscribed to emails", "Visited Booth in Tradeshow", "Email Received", "View in browser link Clicked", "Form Submitted on Website", "Olark Chat Conversation", "Page Visited on Website", "Converted to Lead", and "Email Link Clicked" are most likely to get converted.
# 4.	Country column analysis shows that 90% of the users who got converted and were not converted belong from India, suggesting people from India have a high chance of getting converted.
# 5.	Based on LeadOrigin column, API and Landing Page Submission bring a higher number of leads as well as conversions. Lead Add Form has a very high conversion rate, but the count of leads is not very high. Lead Import and Quick Add Form get very few leads.
# 6.	Specialization column suggests that people enquiring for courses on business management like operations, finance, human resource, market are more likely to get converted.
# 7.	Current Occupation column analysis indicates that unemployed people are more likely to get converted. People who are working are also one of the key leads that can get converted.
# 8.	Last notable activity column analysis shows that people who have interacted with phone or email or who have approached upfront are more likely to get converted.

# #### Numerical Attributes Analysis

# In[74]:


# Checking correlations of numeric values
# figure size
plt.figure(figsize=(10, 8))

# heatmap
sns.heatmap(df.corr(), cmap="RdYlGn", annot=True)
plt.show()


# In[75]:


for i in df.select_dtypes(include=["float64", "int64"]).columns:
    print(f"<=========================={i}================================>")
    plt.figure(figsize=(20, 10))
    sns.boxplot(y=df[i])
    plt.title(i)
    plt.show()


# #### Observations
# 
# 1. There are outliers in the PageViewPerVisit and Total Visit columns.
# 2. These outliers may need to be removed to maintain the stability of the Logistic model.
# 3. No outliers are seen in columns other than the ones mentioned above.

# In[76]:


df = df[df["TotalVisits"] <= 30]


# In[77]:


# Total Visits
# visualizing spread of variable

plt.figure(figsize=(6, 4))
sns.boxplot(y=df["TotalVisits"])
plt.show()


# In[78]:


df = df[df["Page Views Per Visit"] <= 20]


# In[79]:


# Total Visits
# visualizing spread of variable

plt.figure(figsize=(6, 4))
sns.boxplot(y=df["Page Views Per Visit"])
plt.show()


# In[80]:


# Checking correlations of numeric values
# figure size
plt.figure(figsize=(10, 8))

# heatmap
sns.heatmap(df.corr(), cmap="RdYlGn", annot=True)
plt.show()


# In[81]:


sns.pairplot(df)


# In[82]:


# checking percentile values for "Total Visits"

df["TotalVisits"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])


# In[83]:


df["Total Time Spent on Website"].describe(
    percentiles=[0.05, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
)


# In[84]:


create_boxplot_numberical_with_converted(df)


# #### Inference 
# 1. The Total Visits and Page View Per Visit columns have a significant number of outliers. However, after removing the outliers, there was a positive change observed in the correlation matrix.
# 2. There is a strong relationship between the Total Visits and Page View Per Visit columns.
# 3. Individuals who spend more time on the website are more likely to be converted.

# #### Feature Engineering

# In[85]:


for i in labels_with_yesno_values:
    dummies = pd.get_dummies(df[i], prefix=i)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(i, axis=1)


# In[86]:


dummy = pd.get_dummies(
    df[["Lead Origin", "What is your current occupation", "City"]], drop_first=True
)

df = pd.concat([df, dummy], axis=1)
df.drop(
    ["Lead Origin", "What is your current occupation", "City"], axis=1, inplace=True
)


# In[87]:


for i in ["Lead Source", "Last Activity", "Tags", "Last Notable Activity"]:
    dummies = pd.get_dummies(df[i], prefix=i)
    df = pd.concat([df, dummies], axis=1)
    df.drop(i, axis=1, inplace=True)


# In[88]:


dummy = pd.get_dummies(
    df[["Specialization", "What matters most to you in choosing a course", "Country"]],
    drop_first=True,
)

df = pd.concat([df, dummy], axis=1)
df.drop(
    ["Specialization", "What matters most to you in choosing a course", "Country"],
    axis=1,
    inplace=True,
)


# In[89]:


pd.set_option("display.max_columns", None)
df.head()


# In[90]:


df = df.set_index(["Lead Number"])


# ## C.  Data PreProcessing and Model Training

# In[91]:


# ### Import the MinMaxScaler class from the preprocessing module of the scikit-learn library.
# Create an instance of the MinMaxScaler class.
# Select numeric columns from a pandas DataFrame excluding 'Lead Number' and 'Prospect ID' and assign them to the num_cols variable using list comprehension and the select_dtypes method of the DataFrame.
# Apply the scaler to the columns in num_cols using the fit_transform method of the scaler instance.
# Scale the values of the selected columns between 0 and 1 using the MinMaxScaler.

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

num_cols = [
    i
    for i in df.select_dtypes(include=["float64", "int64"]).columns
    if i not in ("Lead Number", "Prospect ID")
]

df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()


# In[92]:


from sklearn.model_selection import train_test_split

# # Import the train_test_split function from the model_selection module of scikit-learn library.
# Assign the 'Converted' column of a pandas DataFrame to variable y.
# Assign the remaining columns of the DataFrame to variable X.
# Split the X and y data into separate training and test sets using the train_test_split function with a 70-30 train-test split ratio and a random state of 42.
# Print the information about the training dataset using the info() method of the X_train DataFrame.

y = df["Converted"]

y.head()

X = df.drop(["Converted", "Prospect ID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, test_size=0.3, random_state=42
)

X_train.info()


# In[93]:


### Evaluate Features by RFE (Recursive Feature Elimination)

cols_filtered_by_rfe = get_rfe_scores(20, X_train, y_train)

print(
    "Columns Which are shortlisted by RFE for Model Training :-",
    sorted(cols_filtered_by_rfe),
)


# In[94]:


vf = get_vif_score(X_train)

vf.sort_values(by="VIF")


# In[95]:


sm1cols = cols_filtered_by_rfe
summary, model, X_train_sm = build_model_glm(X_train, y_train, sm1cols)
summary


# In[96]:


sm2cols = [i for i in sm1cols if i != "Country_France"]
summary, model, X_train_sm = build_model_glm(X_train, y_train, sm2cols)
summary


# In[97]:


sm3cols = [i for i in sm2cols if i != "What is your current occupation_Housewife"]
summary, model, X_train_sm = build_model_glm(X_train, y_train, sm3cols)
summary


# In[98]:


sm4cols = [i for i in sm3cols if i != "Last Activity_Approached directly"]
summary, model, X_train_sm = build_model_glm(X_train, y_train, sm4cols)
summary


# In[99]:


sm5cols = [i for i in sm4cols if i != "Country_Saudi Arabia"]
summary, model, X_train_sm = build_model_glm(X_train, y_train, sm5cols)
summary


# In[100]:


sm6cols = [i for i in sm5cols if i != "Lead Source_Website"]
summary, model, X_train_sm = build_model_glm(X_train, y_train, sm6cols)
summary


# In[101]:


get_vif_score(X_train[sm6cols]).sort_values(by="VIF")


# In[102]:


sm7cols = [i for i in sm6cols if i != "Do Not Email_No"]
summary, model, X_train_sm = build_model_glm(X_train, y_train, sm7cols)
summary


# In[103]:


get_vif_score(X_train[sm7cols]).sort_values(by="VIF")


# In[104]:


sm8cols = [i for i in sm7cols if i != "Lead Source_Traffic Source"]
summary, model, X_train_sm = build_model_glm(X_train, y_train, sm8cols)
summary


# In[105]:


get_vif_score(X_train[sm8cols]).sort_values(by="VIF")


# In[106]:


y_train_pred = model.predict(X_train_sm)
y_train_pred[:10]


# In[107]:


y_train_pred_final = pd.DataFrame(
    {"Converted": y_train.values, "Converted_prob": y_train_pred}
)
y_train_pred_final["Prospect ID"] = y_train.index
y_train_pred_final.head()


# In[108]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[109]:


y_train_pred_final = pd.DataFrame(
    {"Converted": y_train.values, "Converted_prob": y_train_pred}
)
y_train_pred_final["Prospect ID"] = y_train.index
y_train_pred_final.head()


# In[110]:


y_train_pred_final["Predicted"] = y_train_pred_final.Converted_prob.map(
    lambda x: 1 if x > 0.5 else 0
)

# Let's see the head
y_train_pred_final.head()


# In[111]:


from sklearn import metrics

Accuracy, Sensitivity, Specitivity, PRecision, Recall = model_metrics(
    y_train_pred_final.Converted, y_train_pred_final.Predicted
)

print(Accuracy, Sensitivity, Specitivity, PRecision, Recall)


# In[112]:


## The cutoff value of 0.5 has been arbitrarily chosen. In order to determine the best cutoff score,
## we will evaluate all values from 0 to 1.


# In[113]:


numbers = [float(x) / 10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i] = y_train_pred_final.Converted_prob.map(
        lambda x: 1 if x > i else 0
    )
y_train_pred_final.head()


# In[114]:


cutoff_df = pd.DataFrame(columns=["prob", "accuracy", "sensi", "speci"])
from sklearn.metrics import confusion_matrix

num = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for i in num:
    Accuracy, Sensitivity, Specitivity, PRecision, Recall = model_metrics(
        y_train_pred_final.Converted, y_train_pred_final[i]
    )
    cutoff_df.loc[i] = [i, Accuracy, Sensitivity, Specitivity]
print(cutoff_df)


# In[115]:


cutoff_df.plot.line(x="prob", y=["accuracy", "sensi", "speci"])
plt.show()


# In[116]:


#### From the curve above, 0.4 is the optimum point to take it as a cutoff probability.

y_train_pred_final["final_Predicted"] = y_train_pred_final.Converted_prob.map(
    lambda x: 1 if x > 0.4 else 0
)

y_train_pred_final.head()


# In[117]:


y_train_pred_final["Lead_Score"] = y_train_pred_final.Converted_prob.map(
    lambda x: round(x * 100)
)

y_train_pred_final[
    ["Converted", "Converted_prob", "Prospect ID", "final_Predicted", "Lead_Score"]
].head()


# In[118]:


Accuracy, Sensitivity, Specitivity, PRecision, Recall = model_metrics(
    y_train_pred_final.Converted, y_train_pred_final.final_Predicted
)

Accuracy, Sensitivity, Specitivity, PRecision, Recall


# In[119]:


from sklearn.metrics import precision_recall_curve

y_train_pred_final.Converted, y_train_pred_final.final_Predicted
p, r, thresholds = precision_recall_curve(
    y_train_pred_final.Converted, y_train_pred_final.Converted_prob
)

plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[120]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[121]:


plot_res_dist(y_train, y_train_pred)


# ##### To check the performance of the logistic model and assess the data skewness, we will utilize 5-fold cross-validation.The 5-fold cross-validation was performed in order to train the logistic model across different combinations of the available data. This was done to verify the accuracy of the model's predictions and to ensure that the model was robust and able to perform well on new, unseen data. The cross-validation helped to assess the model's performance and also identify any potential issues with data skewness.

# In[122]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)


# In[123]:


from sklearn.tree import DecisionTreeClassifier

decision_tree_model = LogisticRegression()

result = cross_validation(decision_tree_model, X, y, 5)
print(result)


# In[124]:


model_name = "Decision Tree"
plot_result(
    model_name,
    "Accuracy",
    "Accuracy scores in 5 Folds",
    result["Training Accuracy scores"],
    result["Validation Accuracy scores"],
)


# In[125]:


plot_result(
    model_name,
    "Precision",
    "Precision scores in 5 Folds",
    result["Training Precision scores"],
    result["Validation Precision scores"],
)


# In[126]:


plot_result(
    model_name,
    "Recall",
    "Recall scores in 5 Folds",
    result["Training Recall scores"],
    result["Validation Recall scores"],
)


# In[127]:


plot_result(
    model_name,
    "F1",
    "F1 Scores in 5 Folds",
    result["Training F1 scores"],
    result["Validation F1 scores"],
)


# In[128]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[129]:


### Predicition on the Test Data Set


# In[130]:


num_cols = X_test.select_dtypes(include=["float64", "int64"]).columns

X_test[num_cols] = scaler.fit_transform(X_test[num_cols])

X_test = X_test[sm8cols]

X_test.head()


# In[131]:


X_test_sm = sm.add_constant(X_test)

y_test_pred = model.predict(X_test_sm)


# In[132]:


y_test_pred[:10]


# In[133]:


y_pred_1 = pd.DataFrame(y_test_pred)
y_pred_1


# In[134]:


y_test_df = pd.DataFrame(y_test)
y_test_df


# In[135]:


y_test_df["Prospect ID"] = y_test_df.index


# In[136]:


y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[137]:


y_pred_final = pd.concat([y_test_df, y_pred_1], axis=1)
y_pred_final


# In[138]:


y_pred_final = y_pred_final.rename(columns={0: "Converted_prob"})


# In[139]:


y_pred_final.head()


# In[140]:


# Rearranging the columns
y_pred_final = y_pred_final[["Prospect ID", "Converted", "Converted_prob"]]
y_pred_final["Lead_Score"] = y_pred_final.Converted_prob.map(lambda x: round(x * 100))


# In[141]:


y_pred_final["final_Predicted"] = y_pred_final.Converted_prob.map(
    lambda x: 1 if x > 0.4 else 0
)


# In[142]:


y_pred_final.head()


# In[143]:


y_pred_final


# In[144]:


Accuracy, Sensitivity, Specitivity, PRecision, Recall = model_metrics(
    y_pred_final.Converted, y_pred_final.final_Predicted
)

Accuracy, Sensitivity, Specitivity, PRecision, Recall


# In[145]:


draw_roc(y_pred_final.Converted, y_pred_final.Converted_prob)


# In[146]:


plot_res_dist(y_test, y_test_pred)


# ###### Executive Summary:
# 
# The analysis was conducted using a Logistic model to predict the conversion rate for a company. After running the model on the test data, the following observations were made:
# 
# 1. The accuracy for the train data was 83.29% while the sensitivity and specificity were 83.70% and 83.66%, respectively.
# 2. The accuracy for the test data was 84.78%, with sensitivity and specificity values of 83.98% and 85.26%, respectively.
# 
# 3. To verify the accuracy of the model, 5-fold cross-validation was used to train the model across different combinations of data.
# 4. The model performed well in predicting the conversion rate, and the CEO can have confidence in making good decisions based on the model's predictions.
