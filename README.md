# credit-risk-model
## Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretable Models
The Basel II Accord is like a rulebook for banks. It tells them to measure credit risk carefully and explain their decisions. This means we can't just use a model that gives a "yes" or "no" answer without saying why. We need a model that is explainable, documented, and easy to understand — especially in financial environments. This is why banks often use models like Logistic Regression with Weight of Evidence (WoE), which are trusted and transparent.

### 2. The Importance of Proxy Variables
In our dataset, we don’t know exactly who failed to pay back a loan (there’s no "default" column). So, we create a proxy variable — a label that guesses which customers are high-risk based on their behavior. This is necessary to train the model. But if we define this label badly, the model might make poor predictions — such as denying good customers or giving loans to risky ones. That’s why it’s important to create the proxy with care, using methods like RFM (Recency, Frequency, Monetary).

### 3. Trade-offs Between Simple and Complex Models
Simple models like Logistic Regression are easy to explain and accepted in banking, especially when using WoE encoding. But they may not capture complex patterns. Complex models like Gradient Boosting Machines (GBM) are more powerful and accurate but harder to explain. In regulated industries like banking, we must choose models that not only perform well but are also transparent and trustworthy. Often, a balance between interpretability and performance is needed.
