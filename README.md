# Feature-Scaling
Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. It is performed during the data pre-processing to handle highly varying magnitudes or values or units. If feature scaling is not done, then a machine learning algorithm tends to weigh greater values, higher and consider smaller values as the lower values, regardless of the unit of the values. Unscaled data can adversely impact a model’s ability to make accurate predictions.

The dataset we used is "Data_for_Feature_Scaling.csv" where at first we check for the missing value if any then we drop it using dropna(). We calculated mean and median before performing feature scaling. Then we observe that the features have very varying magnitude. In our case, Age and Salary are those features. This is where feature scaling can help us resolve this issue. Feature Scaling can be done in two ways:
#### MinMaxSCaler
MinMaxScaler also known as Normalisation is a sacling technique whereby the values in a column are shifted so that they are bounded between a fixed range of 0 and 1.
The formula for normalisation is as follows:
X_new = (X - X_min) / (X_max - X_min)

#### StandardScaler
StandardScaler also known as Zero-score normalisation is another scaling technique whereby the values in a column are rescaled so that they demonstrate the properties of a standard Gaussian distribution, that is mean = 0 and variance = 1. The formula for standardisation is as follows:
X_new = (X - mean) / std

### Normalisation vs standardisation
*when should we use normalisation and when should we use standardisation?*
The choice between normalisation and standardisation really comes down to the application.

Standardisation is generally preferred over normalisation in most machine learning context as it is especially important when comparing the similarities between features based on certain distance measures. This is most prominent in Principal Component Analysis (PCA), a dimensionality reduction algorithm, where we are interested in the components that maximise the variance in the data.

Normalisation, on the other hand, also offers many practical applications particularly in computer vision and image processing where pixel intensities have to be normalised in order to fit within the RGB colour range between 0 and 255. Moreover, neural network algorithms typically require data to be normalised to a 0 to 1 scale before model training.

*One can always apply both techniques and compare the model performance under each approach for the best result.*

### Which models require feature scaling?
As a matter of fact, feature scaling does not always result in an improvement in model performance. There are some machine learning models that do not require feature scaling.

##### Gradient descent based algorithms
Gradient descent is an iterative optimisation algorithm that takes us to the minimum of a function.
Machine learning algorithms like linear regression and logistic regression rely on gradient descent to minimise their loss functions or in other words, to reduce the error between the predicted values and the actual values.
Having features with varying degrees of magnitude and range will cause different step sizes for each feature. Therefore, to ensure that gradient descent converges more smoothly and quickly, we need to scale our features so that they share a similar scale.

##### Distance-based algorithms
The underlying algorithms to distance-based models make them the most vulnerable to unscaled data.
Algorithms like k-nearest neighbours, support vector machines and k-means clustering use the distance between data points to determine their similarity. Hence, features with a greater magnitude will be assigned a higher weightage by the model.
Also, it is important that we implement feature scaling to our data before fitting them to distance-based algos to ensure that all features contribute equally to predictions.

##### Tree-based algorithms
The tree splits each node in such a way that it increases the homogeneity of that node. This split is not affected by the other features in the dataset.
For that reason, we can deduce that decision trees are invariant to the scale of the features and thus do not require feature scaling.
This also includes other ensemble models that tree-based, for example, random forest and gradient boosting.

## Conclusion
To summarise, feature scaling is the process of transforming the features in a dataset so that their values share a similar scale.
We have learned the difference between normalisation and standardisation as well as 2 different scalers in the Scikit-learn library, MinMaxScaler, StandardScaler.
We also learned that gradient descent and distance-based algorithms require feature scaling while tree-based algorithms do not. 
