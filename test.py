from sklearn.linear_model import LogisticRegression
from feature_selection_ga import FeatureSelectionGA
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_validate

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
x_train, x_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.2)
model = LogisticRegression(multi_class="multinomial", solver="newton-cg")
fsga = FeatureSelectionGA(model,x_train,y_train, verbose=1)
pop = fsga.generate(10)

# compare best individual with random individual
cross_validate(model, X_digits[:, np.array(fsga.best_ind)==1], y_digits,
scoring="balanced_accuracy", cv=5, return_train_score=True)

rand = np.random.choice(X_digits.shape[1], size=sum(fsga.best_ind), replace=False)
cross_validate(model, X_digits[:, rand], y_digits,
scoring="balanced_accuracy", cv=5, return_train_score=True)