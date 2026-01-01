"""
Frontend for the Credit Card Fraud ML Project

Made in 2025 by Chris and Allie using Streamlit!

This script utilizes Streamlit to present a web frontend for our
ML project we created in Fall 2025 for the OSU-AI Club.

The outline of this frontend includes these sections:

1. AUTHORS + AUTHOR Links
2. Mission statement/Introduction to the problem
3. Our data + its limitations and strengths
4. Our choice of model + design decisions
5. Performance metrics
6. Reflections
7. Bibliography
"""

import pandas as pd
import streamlit as st

# Sidebar Page Navigation
with st.sidebar:
    st.title("Page Navigation")
    st.markdown(
        """
        [Credit Card Fraud Classification Project](#credit-card-fraud-classification-project)

        [What is this?](#what-is-this)

        [Mission statement + problem introduction](#mission-statement-problem-introduction)

        [Our data with strengths + weaknesses](#our-data-with-strengths-weaknesses)

        [Our choice of models + design decisions](#our-choice-of-models-design-decisions)

        [Our performance metrics](#our-performance-metrics)

        [Our reflections](#our-reflections)

        [Bibliography](#bibliography)
        """
    )

st.set_page_config(
    page_title="OSU AI F25 Credit Card Fraud Project",
    page_icon=":robot:",
)

# 1. AUTHORS + AUTHOR Links
st.title(
    """
    Credit Card Fraud Classification Project
    """
)

st.subheader(
    """
    Made for OSU AI Club F25 Project Workshop
    """
)

st.write(
    """
    ## What is this?
    This is a presentation (powered by streamlit) of our credit card 
    fraud ML project for [OSU's AI Club Project Workshop Fall 2025](https://www.osu-ai.club/project-workshop/).

    The Project Workshop hosted by the [AI Club at OSU](https://www.osu-ai.club/) 
    was a way for us
    to get our feet wet with the ML model life cycle. Thus, we aimed
    to build a beginner-friendly ML portfolio project from conception 
    to deployment. This includes: collecting data, data-preprocessing, 
    model training, hyperparameter tuning, performance measurments,
    and some form of project hosting for our portfolios.

    We aim to share our own discoveries throughout, as well as
    any questions for further inquiry!

    **Authors: Undergraduate OSU CS students, Chris and Allie**
    """
)

# <place links here>

st.page_link(
    page="https://github.com/cknell47/ccfd",
    label="Project Repository",
    help="https://github.com/cknell47/ccfd",
)

st.write(
    """
    ### Tech stack
    - Python
        - Jupyter Notebook
        - Scikit-learn
        - Y-Data Profiling
        - Pandas
        - NumPy
        - Matplotlib
        - Streamlit
    - git/GitHub
    - ssh
    """
)

# 2. Mission statement/Introduction to problem
st.write(
    """
    ## Mission statement + problem introduction

    Reiterating what was mentioned in the foreword.. Our goal is to
    create our first ML project. Based on the presentations given by
    the OSU AI Club, we thought it would be viable to try
    a classification problem. This means we won't have any confidence
    probabilities like we might have if we chose to do regression
    instead. Eventually, we settled on this problem
    statement:

    **"Detect fraudulent credit-card transactions with Kaggle Credit 
    Card Fraud Dataset in real time while striking a balance between 
    actual fraud prevention with customer friction."**

    **Illustrative example**: It's one thing if someone doesn't get 
    their Starbucks coffee one morning. However, we want to avoid the case
    of a parent with a family of four is stuck at a Walmart 
    self-checkout because an ML model makes a wrong decision over 
    groceries being bought with a slightly higher transaction due to 
    *gasps* inflation. 

    Therefore, we believe this story illustrates a need for emphasis on
    a minimization of false-positives and false-negatives to minimize
    user friction. Essentially, optimized model performance!
    """
)

# 3. Our data + its limitations and strengths
df = pd.read_csv("./data/creditcard.csv")
st.write(
    """
    ## Our data with strengths + weaknesses

    We decided between a few datasets when deciding how to source
    data for this project. Not all of them were great, *including the
    one we settled on*. We made the final decision by balancing
    accessibility (including legal use), with applicability to
    our problem.

    The exact details of our choice and how we evaluated alternatives
    is in the README.md of the project [here](https://github.com/cknell47/ccfd).

    We eventually selected [this dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 
    hosted by Kaggle. The data's story is this:

    In September 2013, transactions over two days by European card
    holders were measured and recorded. There were 284,807 transactions
    total, with 492 fraud cases identified. The class is denoted with 
    a 0 or 1 denoting a fraud case as 1. The dataset only includes
    numerical data, which has been obtained from the result of a "PCA"
    transformation to achieve confidentiality. 

    As a result of the PCA transformation on the features of the 
    dataset, we had a hard time making design decisions 
    about our model's classifications beyond performance comparisons. 
    We were also limited because, since the PCA transformation is 
    secret, we can't, for instance, classify new data beyond the dataset.

    It's also worth pointing out that there is a massive class
    imbalance in the dataset (98.2% not-fraud) which we needed to
    accomodate in our model training.

    Accounting for these weaknesses, we essentially decided to 
    maximize our models' performance directly on this dataset.
    We wrote off any potential use of this in real-world, but we
    felt that was okay because this is mainly a learning project.

    <place information here>

    ### Interpretations of Y-Data Profiling on dataset

    I believe it is worth mentioning here we used the module Y-Data
    for data exploration help. From this, we learned there are some
    duplicate rows in the dataset, the relative imbalance, as well
    as the fact that V10-12, V14, and V16-18 were highly correlated
    with class. This information was used to reduce training time and
    improve final performance.
    """
)

st.divider()

st.write(
    """
    Here is the raw data as a dataframe of the Kaggle dataset.
    
    <This is the source link.> It has distributions and other info
    we elected to leave out from this presentation.
"""
)
st.space()
st.write(df)

# 4. Our choice of model + design decisions
st.write(
    """
    ## Our choice of models + design decisions

    Based on the club presentation in Week 4 of the term, we decided
    to pick 2 models to start to compare performance.

    These were SGDClassifier, which was picked as a result of the
    model flowchart hosted by Scikit-learn, and Random Forest
    Classifier, which was selected because supposedly it is good
    for tabular data (which we have).

    <place information here>
    """
)

# 5. Performance metrics
st.write(
    """
    ## Our performance metrics

    <place information here>
    """
)

# 6. Reflections
st.write(
    """
    ## Our reflections

    <place information here>
    """
)

# 7. Bibliography
st.write(
    """
    ## Bibliography

    <place information here>

    Kaggle dataset: [link]()

    """
)
