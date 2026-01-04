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
    model training, hyperparameter tuning, performance measurements,
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

    ### Interpretations of Y-Data Profiling on dataset

    I believe it is worth mentioning here we used the module Y-Data
    for data exploration help. From this, we learned there are some
    duplicate rows in the dataset, the relative imbalance, as well
    as the fact that V10-12, V14, and V16-18 were highly correlated
    with class. This information was used to reduce training time and
    improve final performance.
    """
)

st.write(
    """
    Here is the link for the raw data of the Kaggle dataset. It has distributions and other info
    we elected to leave out from this presentation since the file is too large on GitHub.
    """
)

st.page_link(
    page="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
    label="Kaggle Credit Card Fraud Dataset",
    help="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
)

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

    We divied the work on each model to each of us, Allie taking SGDClassifier
    and Chris taking the RandomForestClassifier.

    ### SGD Classifier decisions

    I'll be organizing my design decisions for the ultimate SGDClassifier performance
    with the phases mentioned in the introduction: data-preprocessing, model training, and hyperparameter
    tuning.

    #### Data-preprocessing

    Based on the club presentations and from research online, I tried to use Standard scaling via Scikit-learn
    on our data to improve the model performance. This ended up having a small performance boost in True Positive fraud cases.

    I also tried to use upsampling of the minority class (in this case, the fraud cases) to match the majority class. This seemed
    to decrease performance however, as well as significantly increased training time, so I dropped the strategy.
    
    #### Model training

    From the result of Y-data profiling, it was found that specific features like V10-12, V14, V16-18 were "highly correlated"
    with the "Class" feature. I took that to imply that those were the most significant features, and in my experience it seemed like
    selecting those specific features (excluding others) increased performance slightly.

    To train the model and prevent data leakage, I did a train-test split of 80-20. This seemed like a reasonable choice, but FWIW I never tried
    weighting the values differently and comparing performance. 

    I think it is important, especially in an unbalanced dataset like ours,
    to use the stratify parameter in the train_test_split() function. I stratified based on the "Class" feature so that the train
    and test sets both had a fair proportion of fraud and non-fraud cases. This avoids the case where the train or test set has way more fraud
    cases than the other set, skewing the performance metrics.

    #### Hyperparameter tuning

    Based on some GridSearchCV results in a prior experiment on the model and dataset, I decided to choose loss='square_hinge' and penalty='l1'.
    This slightly improved performance.
    
    Admittedly, there wasn't much going into this decision, aside from the fact that testing parameters other than "penalty" and "loss" was leading
    to some errors like "eta0 < 0" for the model. This was potentially caused by changes in the "learning_rate" parameter, but I never figured out how
    to resolve the issue other than keep the default learning rate.

    The specific details of that can be found in the SGDClassifier notebook in the repository under "Hyperparameter Tuning".
    """
)

# ===

# 5. Performance metrics
st.write(
    """
    ## Our performance metrics

    Finally, moving onto our metrics. We decided on a few general metrics based on the club presentations
    and our own research into what would be reasonable for our problem statement. Especially considering our
    priority on minimizing false positives and negatives, we figured a confusion matrix and AUPRC was reasonable.
    Also the use of cross-validation seemed valuable generally across projects to protect against specific data point
    overfitting etc.
    
    The reason for any number discrepancy is because for SGDClassifier we dropped duplicates and didn't for RandomForestClassifier.

    The format is SGDClassifier first, then RandomForest beneath it.
    """
)

st.write(
    """
    ### Classification Report and Confusion Matrix
    """
)

st.image(
    "./data/SGDClass-Images/class-report-and-confusion-matrix.png", width="stretch"
)

st.image("./data/RandomForest-Images/classification-report.png", width="stretch")


st.image("./data/RandomForest-Images/confusion-matrix.png", width="stretch")

st.write(
    """
    ### Seaborn Confusion Matrix Heatmap
    """
)

st.image("./data/SGDClass-Images/sgd-seaborn.png", width="stretch")

st.image("./data/RandomForest-Images/heatmap.png", width="stretch")

st.write(
    """
    ### Cross Validation
    """
)

st.image("./data/SGDClass-Images/cross-validation.png", width="stretch")

# st.image("./data/RandomForest-Images/", width="stretch")

st.write(
    """
    ### PRC Curve
    """
)

st.image("./data/SGDClass-Images/auprc.png", width="stretch")

st.image("./data/RandomForest-Images/auprc.png", width="stretch")

st.write(
    """
    SGDClassifier Area under PRC curve = 0.7333745190569216

    RandomForestClassifier Area under PRC curve = 0.8751277004172419
    """
)

# ===

# 6. Reflections
st.write(
    """
    ## Our reflections

    It seems like RandomForestClassifier was able to perform better on the dataset generally with
    less false negatives and positives and a higher AUPRC.

    Given that this was a first project ever working with ML models directly, this was challenging!
    Especially juggling the project with school and other obligations. I wanted to thank my partner
    Chris on this project for making it socially fulfilling as well as for fueling the educational
    dialectic we had describing why and how for certain parts of the architecture in the project.

    In terms of the future considerations, I think for another project I'd probably use a dataset where
    none of the features are anonymized. That significantly limited the scope of the project (especially in deployment) but in many ways
    that was helpful for keeping scope creep manageable. Potentially, like limiting an artist's palette to reduce
    overwhelm. As a result, we got to focus on how and why we do each step in the process of dataset selection,
    data preprocessing + exploration, hyperparameter tuning, model training and selection, strategies for working around data leakage,
    and how to measure performance accounting for potential overfitting (among others).

    Overall, I'm relatively proud of this project. I want to thank the OSU AI club for their
    significant help and structure for learning about the fundamental process of training an ML
    model. The tutorials and presentations were very helpful with direction and I'm looking forward
    to Winter term! I learned a lot about what not to do especially. Getting through this first
    project feedback cycle is always extremely challenging for me and I'm excited for the potential
    in the next one :).
    """
)

# 7. Bibliography
st.write(
    """
    ## Bibliography

    ### Placing significant resources in our development of the project here

    Kaggle dataset: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
    
    OSU AI Website (slides, tutorials, project guidance): [OSU-AI](https://www.osu-ai.club/)

    Handling imbalanced datasets: [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/handling-imbalanced-data-for-classification/)

    AUPRC Resource and implementation: [StackOverflow](https://stackoverflow.com/questions/71934403/how-to-get-the-area-under-precision-recall-curve)

    Scikit-learn estimator flow chart we used for SGDClassifier: [Scikit-learn documentation](https://scikit-learn.org/stable/machine_learning_map.html)

    Resource for implementation of RandomForestClassifier and SGDClassifier:

    RandomForestClassifier: [YouTube](https://www.youtube.com/watch?v=AZNrn9ihZkw)
    SGDClassifier: [YouTube](https://www.youtube.com/watch?v=UWe_oF2lh9g)

    Understanding git merging vs rebasing after a significant version history error: [Atlassian](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)

    #### Resources used for RandomForest.ipynb

    Largely, these are code snippets and video tutorials that Chris used for the RandomForestClassifier pipeline.

    [Kaggle Code Snippet for Confusion Matrix](https://www.kaggle.com/code/bernabas/random-forest-with-confusion-matrix)

    Web search Brave AI Conversations

    [How to remove nan errors in Python](https://search.brave.com/search?q=how+to+remove+nan+errors+in+python&summary=1&conversation=a53893a5a843463cc415f5)

    [Cross Validation Mean Scores](https://search.brave.com/search?q=how+to+get+cross+validation+and+mean+scores+for+random+forest+classifier&summary=1&conversation=8946dd8b8577727f04c60f)

    YouTube video

    [Random Forest GridSearchCV](https://www.youtube.com/watch?v=c4mS7KaOIGY)

    Google AI Conversation

    [Python AUPRC](https://www.google.com/search?q=python+auprc&rlz=1C1ONGR_enUS988US988&oq=python+auprc&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIICAEQABgWGB4yBwgCEAAY7wUyBwgDEAAY7wUyCggEEAAYgAQYogTSAQgzNDQ4ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
    """
)
