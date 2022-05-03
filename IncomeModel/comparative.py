# LINEAR MODELS
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor

# NEIGHBORS
from sklearn.neighbors import KNeighborsRegressor

# SVM
from sklearn.svm import SVR

# TREES AND ENSEMBLE METHODS
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor


def cars_regression_models(regression_model):
    piped_regressor = make_pipeline(preprocessor, regression_model)
    return piped_regressor

# Here is an example of a pipelined regressor

from sklearn.neighbors import KNeighborsRegressor
cars_regression_models(KNeighborsRegressor())

models = [LinearRegression(),
          Ridge(),
          Lasso(),
          ElasticNet(),
          SGDRegressor(),
          KNeighborsRegressor(),
          SVR(kernel = "linear"),
          SVR(kernel = "poly", degree = 2),
          SVR(kernel = "poly", degree = 3),
          SVR(kernel = "rbf"),
          DecisionTreeRegressor(),
          RandomForestRegressor(),
          AdaBoostRegressor(),
          GradientBoostingRegressor()
]

from sklearn.model_selection import train_test_split

% % time

X_train, X_test, y_train, y_test = train_test_split(X, y)
different_test_scores = []

for model_name, model in zip(models_names, models):
    temp_piped_regressor = cars_regression_models(model)
    temp_piped_regressor.fit(X_train, y_train)
    different_test_scores.append(temp_piped_regressor.score(X_test, y_test))

comparing_regression_models_cars = pd.DataFrame(list(zip(models_names, different_test_scores)),
                                                columns=['model_name', 'test_score'])

round(comparing_regression_models_cars.sort_values(by="test_score", ascending=False), 2)

####Crossvalidated

# %%time

# different_test_scores_cv = []

# for model_name, model in zip(models_names, models):

#     temp_piped_regressor = cars_regression_models(model)
#     different_test_scores_cv.append(cross_val_score(temp_piped_regressor, X, y).mean())

# comparing_regression_models_cars_cv = pd.DataFrame(list(zip(models_names, different_test_scores)),
#                                                 columns = ['model_name', 'cross_val_score'])

# round(comparing_regression_models_cars_cv.sort_values(by = "cross_val_score", ascending = False),2)