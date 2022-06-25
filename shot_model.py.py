import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    r2_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
import snowflake.connector


FEATURE_LIST = [
    "ENDGAMECLOCK",
    "PERIOD",
    "TOTALGAMECLOCK",
    "LOCATIONX",
    "LOCATIONY",
    "DISTANCE",
    "SHOTCLOCK",
    "CONTESTED",
    "CLOSESTDEFDIST",
    "SHOTNUMBER",
    "DRIBBLESBEFORE",
    "SHOOTERSPEED",
    "SHOOTERVELANGLE",
    "DISTANCExCLOSESTDEFDIST",
]


def create_parser():
    """
    Create command-line argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="rf",
        help="Type of shot quality model to build",
    )
    parser.add_argument(
        "--trees",
        type=int,
        required=False,
        default=None,
        help="Number of decision trees in random forest model",
    )
    parser.add_argument(
        "--depth",
        type=int,
        required=False,
        default=None,
        help="Maximum depth of random forest model",
    )
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="Whether to save model predictions and plots",
    )
    return parser


def query_data():
    """
    Query raw data from Snowflake database

    Returns
    -------
    data: pd.DataFrame
        Queried data with the columns to be used in the model training process
    """
    ctx = snowflake.connector.connect(
        user="candidate_grace_peng",
        password="nvg*dbr9xft3BHK9wft",
        account="ena18838.us-east-1",
        role="CANDIDATE",
        warehouse="WOLVES_PUBLIC_WAREHOUSE",
    )

    cur = ctx.cursor()
    cur.execute(
        """
        SELECT 
            shots.ID,
            shots.player_name,
            shots.qsp,
            shots.qsq,
            shots.three,
            shots.fouled,
            shots.period,
            shots.season,
            shots.outcome,
            shots.distance,
            shots.shottype,
            shots.contested,
            shots.locationx,
            shots.locationy,
            shots.shotclock,
            shots.endgameclock,
            shots.closestdefdist,
            shots.closestdefapproach,
            shots.dribblesbefore,
            shots.shooterspeed,
            shots.shootervelangle,
            720 - shots.endgameclock + (shots.period - 1) * 720 AS totalgameclock,
            ROW_NUMBER() OVER (PARTITION BY player_name, gameId ORDER BY game_date) AS shotnumber
        FROM wolves_public.analyst_project.shots
        WHERE shots.shotclock IS NOT NULL and shots.shotclock <= 24
        """
    )
    data = cur.fetch_pandas_all().reset_index()
    cur.close()

    print(len(data))
    return data


def train_regression_model(features, outcomes):
    """
    Train logistic regression model using feature set and outcomes as target metric

    Parameters
    -------
    features: pd.DataFrame
        Data table representing the FEATURE_LIST metrics in the shot data
    outcomes: pd.Series
        Data column representing the outcome of the shot, 0 or a miss and 1 for a make

    Returns
    -------
    lr: LogisticRegression
        Trained regression classifier model with L2 penalty
    """
    lr = LogisticRegression(penalty="l2", max_iter=500, random_state=0, solver="lbfgs")
    lr.fit(features, outcomes)
    return lr


def train_random_forest_model(X_train, y_train, trees, depth):
    """
    Train random forest model with training data split from original dataset, either with 
    user specified hyperparameters for the model, or with a grid search 

    Parameters
    -------
    X_train: pd.DataFrame
        A subset of the features from the train-test split
    y_train: pd.Series
        A subset of the outcomes from the train-test split
    trees: int
        The number of trees to use to train the model
    depth: int
        The maximum depth of the tree

    Returns
    -------
    classifier: RandomForestClassifier
        Trained random forest model with turned parameters or the given parameters
    """
    if trees is None or depth is None:
        rf = RandomForestClassifier()
        param_grid = {"n_estimators": [200, 300, 400, 500], "max_depth": [2, 3, 4, 5]}
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        classifier = RandomForestClassifier(
            n_estimators=grid_search.best_params_["n_estimators"],
            max_depth=grid_search.best_params_["max_depth"],
        )
    else:
        classifier = RandomForestClassifier(n_estimators=trees, max_depth=depth)
    classifier.fit(X_train, y_train)

    return classifier


def print_eval_metrics(model, features, outcomes, save):
    """
    Make predictions with input features and calculate model evaluation metrics

    Parameters
    -------
    model: 
        Which model to evaluate - either lr or rf
    features: pd.DataFrame
        Data table representing the FEATURE_LIST metrics in the shot data
    outcomes: pd.Series
        Data column representing the outcome of the shot, 0 or a miss and 1 for a make
    save: bool
        Whether to save the calibration plot

    Returns
    -------
    preds: np.array
        Model predictions for the given set of features
    """
    binary_preds = model.predict(features)
    preds = model.predict_proba(features)[:, 1]
    rmse = np.sqrt(mean_squared_error(outcomes, preds))
    print("Accuracy: ", accuracy_score(binary_preds, outcomes))
    print("RMSE: ", round(rmse, 5))
    print("R2: ", r2_score(outcomes, preds))
    print("Brier loss: ", brier_score_loss(outcomes, preds))

    x, y = calibration_curve(outcomes, preds)
    plt.plot(x, y)
    plt.plot([0, 1])
    plt.xlabel("Mean predicted probaiblity")
    plt.ylabel("Fraction of positives")
    if save:
        plt.savefig("calibration_curve.png")

    return preds


def get_analysis_tables(data, pred_file, players_file, save):
    """
    Modify dataset with predictions appended to find difference between expected and observed makes
    for each player, as well as average shot probabilities for each shot type

    Parameters
    -------
    data: pd.DataFrame
        The queried, processed dataset with the predictions appended
    pred_file: str
        Name of the predictions file to save as
    players_file: str
        Name of the players ranking in the delta of expected and observed makes file to save as
    save: bool
        Whether to save the tables as CSVs
    """
    data["diff"] = (data["OUTCOME"] - data["pred"]) * (data["THREE"] + 2)
    players = (
        data.groupby("PLAYER_NAME").agg({"ID": "count", "diff": "mean"}).sort_values(by="diff")
    )
    shot_types = (
        data.groupby("SHOTTYPE").agg({"ID": "count", "pred": "mean"}).sort_values(by="pred")
    )
    if save:
        data.to_csv(f"{pred_file}.csv")
        players.to_csv(f"{players_file}.csv")


def plot_features(data, x_name, y_name, plot_title, plot_file, save):
    """
    Plot top important features in the model

    Parameters
    -------
    data: pd.DataFrame
        The queried, processed dataset with the predictions appended
    x_name: str
        Name of feature column to plot on the x-axis
    y_name: str
        Name of feature column to plot on the y-axis
    plot_title: str
        Title of the plot
    plot_file: str
        Filename to save the plot as
    save: bool 
        Whether to save plot figure
    """
    plt.scatter(
        data[x_name][::100],
        data[y_name][::100],
        marker=".",
        s=50,
        linewidths=4,
        c=data["pred"][::100],
        cmap=plt.cm.coolwarm,
        alpha=0.5,
    )
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(plot_title)
    if save:
        plt.savefig(f"{plot_file}_plot.png")


def main():
    parser = create_parser()
    args = parser.parse_args()
    print(args)

    data = query_data()

    ### PREPROCESSING ###
    data["DISTANCExCLOSESTDEFDIST"] = data["DISTANCE"] * data["CLOSESTDEFDIST"]
    data["CLOSESTDEFAPPROACH"] = (
        data["CLOSESTDEFAPPROACH"][0].strip("[]").replace(" ", "").split(",")[3]
    )
    training_data = data[np.logical_or(data.FOULED == 0, data.OUTCOME == 1)]

    outcomes = training_data["OUTCOME"]
    features = training_data[FEATURE_LIST]

    ### RUN LOGISTIC REGRESSION MODEL ###
    if args.model_type == "lr":
        lr = train_regression_model(features, outcomes)

        ### PRINT EVALUATION METRICS ###
        for i, j in sorted(zip(features.columns, lr.coef_[0]), key=lambda x: x[1], reverse=True):
            print(i, "{0:.10f}".format(math.e ** (j)))

        print_eval_metrics(lr, features, outcomes, args.save)

    ### RUN RANDOM FOREST MODEL ###
    if args.model_type == "rf":
        X_train, X_test, y_train, y_test = train_test_split(features, outcomes, test_size=0.2)
        classifier = train_random_forest_model(X_train, y_train, args.trees, args.depth)

        ### PRINT EVALUATION METRICS ###
        preds_train = print_eval_metrics(classifier, X_train, y_train, args.save)
        preds_test = print_eval_metrics(classifier, X_test, y_test, args.save)

        importances = {}
        for i, j in sorted(
            zip(features.columns, classifier.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        ):
            importances[i] = "{0:.10f}".format(j)

        print(
            (pd.DataFrame.from_dict(importances, orient="index", columns={"importance"})).to_markdown()
        )

        ### CREATE ANALYSIS TABLES ###
        X_train_temp = pd.concat([X_train.reset_index(), pd.Series(preds_train)], axis=1).rename(
            columns={0: "pred"}
        )
        X_test_temp = pd.concat([X_test.reset_index(), pd.Series(preds_test)], axis=1).rename(
            columns={0: "pred"}
        )
        predictions = pd.concat([X_train_temp, X_test_temp])
        compare = predictions.reset_index()[["index", "pred"]].merge(
            training_data.drop("index", axis=1).reset_index(), on="index"
        )
        get_analysis_tables(compare, "non_foul_predictions", "non_foul_over_under_performers", args.save)

        # If we want to also make predictions for shots where a foul occurred and shot missed
        predictions = classifier.predict_proba(data[FEATURE_LIST])[:, 1]
        all_data = pd.concat([data, pd.Series(predictions)], axis=1).rename(columns={0: "pred"})
        get_analysis_tables(all_data, "all_predictions", "all_over_under_performers", args.save)

        ### CREATE ANALYSIS PLOTS ###
        # Plot every 100 points to not overload
        plot_features(
            compare,
            "DISTANCE",
            "CLOSESTDEFDIST",
            "Shot Distance and Defender Distance Effects on Predicted Make Probability",
            "dist_def",
            args.save,
        )
        plot_features(
            compare,
            "LOCATIONX", 
            "LOCATIONY", 
            "Shot Location Effect on Predicted Make Probability", 
            "location",
            args.save,
        )


if __name__ == "__main__":
    main()


### CALIBRATION ###
# from sklearn.calibration import CalibratedClassifierCV
# calib_model = CalibratedClassifierCV(classifier, method="sigmoid", cv=5)
# calib_model.fit(X_train, y_train)
# prob = calib_model.predict_proba(X_test)[:,1]
