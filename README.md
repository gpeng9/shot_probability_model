# Shot Probability Model

## Project Contents

- `all_predictions.csv` - includes `pred` column appended to the original queried data with all data except the rows with null values and unrealistic shot clock values. (This file was taken out of the zip because it was too large to send. It should be fairly easy to reproduce with the code.)
- `all_over_under_performers.csv` - using the above data, in particular the `diff` column, finds the difference between the true and expected shot outcome, and averaged across each player to find the over- and under-performers.
- `non_foul_predictions.csv` - same as `all_predictions`, but without shots where a foul occurred and the shot was missed. (This file was taken out of the zip because it was too large to send. It should be fairly easy to reproduce with the code.)
- `non_foul_over_under_performers.csv` - same as `all_over_under_performers`, but without shots where a foul occurred and the shot was missed.
- `calibration_curve.png` - plot of the calibration of the tuned random forest model with 200 trees and max depth of 5.
- `dist_def_plot.png` - plot of the tuned random forest model predictions with the distance on the x-axis and closest defender distance on the y-axis. The data does not include shots where a foul occurred and the shot was missed. Only every 100th point was plotted to not overload the plot.
- `location_plot.png` - plot of the tuned random forest model predictions with the x location of the shot on the x-axis and y location on the y-axis. The data does not include shots where a foul occurred and the shot was missed. Only every 100th point was plotted to not overload the plot.
- `shot_model.py` - the script that reproduces the above results. There is an argument parser for the user to use:
  - `--model_type` - which model to use, either "rf" (default) or "lr"
  - `--trees` - number of trees to use in the random forest model corresponding to the number of estimators argument in the sklearn module. If not specified, a grid search will be run to tune the parameter
  - `--depth` - depth of the random forest model corresponding to the max depth argument in the sklearn module. If not specified, a grid search will be run to tune the parameter
  - `--save` - whether to save the output files and plots locally, described above

### Model Features

|                         |   importance |
|:------------------------|-------------:|
| DISTANCE                |  0.348492    |
| DISTANCExCLOSESTDEFDIST |  0.175007    |
| CONTESTED               |  0.120364    |
| LOCATIONX               |  0.114624    |
| CLOSESTDEFDIST          |  0.0838595   |
| LOCATIONY               |  0.0407059   |
| SHOOTERSPEED            |  0.0379018   |
| SHOTCLOCK               |  0.0315767   |
| DRIBBLESBEFORE          |  0.0226318   |
| ENDGAMECLOCK            |  0.0138671   |
| SHOOTERVELANGLE         |  0.00883905  |
| SHOTNUMBER              |  0.00103346  |
| TOTALGAMECLOCK          |  0.0010161   |
| PERIOD                  |  8.25458e-05 |

- `DISTANCExCLOSESETDEFDIST` - Intuitively thinking about this, even though a farther distance probably leads to a lower prediction, actually when the closeset defender is far away as well, the prediction becomes slightly higher. Same with closer shots, which in general lead to higher make probabilties, but a defender closer will cause the prediction to be lower.
- `TOTALGAMECLOCK` - While `PERIOD` and `ENDGAMECLOCK` both exist, the latter resets with each quarter. Combining the two into a new feature quantifies how much total time has elapsed in the game, thinking maybe as the game goes on, make probabilities go down due to fatigue.
- `SHOTNUMBER` - Similarly, as the game goes on, a player's tenth shot might be more fatigued than his third shot. The importance was not high, however, leading me to think maybe some players heat up as they go, or do not play long enough for their range of shots to be much different in fatigue (like for role players).

### Extra Thoughts

We see the resulting leaderboard on all data hurts players who get fouled a lot more (i.e. Lebron) since including those fouled shots are all misses, even though they could be easier shots. It makes sense to look at the one that does not include fouled, missed shots. The leaderboard in general, filtering out small sample players over the 7-8 years, we see the top performers, besides the obvious star players, are sharp shooters and efficient big men.

The `dist_def_plot` is interesting to me, as it shows the interaction term between DISTANCE and CLOSESETDEFDIST. Even though the shot starting from 5 feet are blue, a lighter blue (higher make prediction) exists in the area where the closest defender distance is greater. And for close distance shots, while they're mostly red, the lighter red (lower make prediction) exists where the closest defender distance is lesser. In fact, all the points where the closest defender is super far, and the shot distance is super close, are probably fast break layups.

The `location_plot` was cool to visualize the heat map of the shot chart and seeing where the shot probability starts to decrease. It provides a nice sanity check, and helped me realize that LOCATION_X referred to the length of the court and not the width, which made the feature importances rank make more sense.

## Future Work

Random forests are an easy first model to implement that also provides fast predictions, but it can be improved upon. A better model might be a boosting model, or a deep neural net, and then to add the interpretability, we could somehow build a framework on top to linearly combine features that would provide more transparency to stakeholders and be easier to explain. There could also be more EDA done and thought put into feature engineering, such as whether defender approach could be used to learn at what point the defender starts to affect or distract the shooter. Obtaining more features like the score, or whether it was a home or away game could help as well.

Something that is slightly different from what this project is asking, would be adding in player random effects or team/lineup effects that could impact what would be considered a good shot in the context of the matchup (both shooter vs closest defender, and assisting team vs defending team), which could be different flavors of the shot probability model. While this model measures shot difficulty and considers 3s relatively difficult, the shot quality measure could be different for 3s, since they are so common in the NBA now, and considered more "quality" shots than they were years ago. Also, instead of ignoring data with fouls, since drawing fouls is somewhat a skill, it would be interesting to incorporate that, maybe as a separate model that relates to offensive ability as a whole.

