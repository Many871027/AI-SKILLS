# EDA and Statistical Best Practices
When analyzing the dataset prior to training, adhere to the following rigorous standards:

1.  **Univariate Analysis:**
    * Examine the distribution of the target variable. If minority class < 20%, trigger SMOTE.
    * Check for skewness in numerical features. If skewness > 1 or < -1, recommend log transformation before the MinMaxScaler.
2.  **Bivariate & Multivariate Analysis:**
    * Check for multicollinearity. If two features have a Pearson correlation > 0.85, recommend dropping one to simplify the Naive Bayes assumption of feature independence.
3.  **Bayesian Probability:**
    * Remember that Naive Bayes relies on prior probabilities. Ensure the training set reflects the real-world base rates unless specific cost-sensitive learning is required.