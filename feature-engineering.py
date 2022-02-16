import gzip
import pandas as pd
import numpy as np
import re
import sys
from collections import defaultdict

def main():
    """
    Load the UCI "ADULT" Census income data set and perform
    feature engineering.
    """

    features = Features(
        training_data=pd.read_csv(sys.argv[1], na_values="?"),
        testing_data=pd.read_csv(sys.argv[2], na_values="?"),
        csc_file=sys.argv[4]
    )

    # Expand categorical values into dummy features
    features.categorical("workclass")
    features.categorical("education")
    features.categorical("marital_status")
    features.categorical("occupation")
    features.categorical("relationship")
    features.categorical("race")
    features.categorical("sex")
    features.categorical("native_country")

    # Standardize continuous features and add higher order terms
    features.continuous("age", squared=True, cubed=True)
    features.continuous("hours_per_week", squared=True, cubed=True)

    # Expand hurdle features into extensive and intensive margins.
    features.hurdle("capital_gain", log=True)
    features.hurdle("capital_loss", log=True)

    # Drop unused features
    features.drop("education_num")

    # Add all pairwise interactions
    features.interactions()

    # Close compressed sparse column file
    features.close_csc()

    # Write subset/outcome/weights variables to csv file
    features.to_csv(["subset", "salary_50k", "fnlwgt"], sys.argv[3])


class Features(object):

    def __init__(self, training_data, testing_data, csc_file):
        """
        Combine separate data frames for training/testing into
        a single data frame with a subset indicator.
        """
        assert "subset" not in training_data.columns
        training_data["subset"] = "TRAIN"
        assert "subset" not in testing_data.columns
        testing_data["subset"] = "TEST"
        assert (training_data.columns == testing_data.columns).all()
        self.X = pd.concat([training_data, testing_data], ignore_index=True)
        self.training_mask = (self.X.subset == "TRAIN")
        self.outstanding_features = set(self.X.columns)
        self.feature_map = defaultdict(list) # Tracks sets of features for determining pairwise interactions
        self.csc_file = gzip.open(csc_file, "wt")
        self.csc_file.write("#csc start nrow={}\n".format(len(self.X)))
        self.csc_columns = set()

    def write_sparse_column(self, name, col=None):
        """
        Write a modified column as a sparse vector to the output file.
        """
        assert name not in self.csc_columns, name
        if col is None:
            col = self.X[name]
        if len(col.loc[self.training_mask].unique()) == 1:
            print(f"WARNING: dropping empty column {name}")
        else:
            col = col.astype(float).to_numpy()
            index = col.nonzero()
            self.csc_file.write(f"column {name}\n")
            self.csc_file.write(" ".join(str(i) for i in index[0]))
            self.csc_file.write("\n")
            self.csc_file.write(" ".join("{:g}".format(x) for x in col[index]))
            self.csc_file.write("\n")
            self.csc_columns.add(name)

    def drop(self, feature_name):
        """
        Drop a feature.
        """
        del self.X[feature_name]
        self.outstanding_features.remove(feature_name)

    def categorical(self, feature_name):
        """
        Expand a categorical feature into dummies, dropping the most frequent
        category to avoid collinearity.
        """
        assert feature_name in self.outstanding_features, feature_name
        self.X.loc[:, feature_name].fillna("MISSING", inplace=True)
        values = self.X.loc[self.training_mask, feature_name].value_counts()
        for value in values[1:].index:
            dummy_name = sanitize_name(f"{feature_name}_{value}")
            assert dummy_name not in self.X.columns, dummy_name
            self.X[dummy_name] = (self.X[feature_name] == value).astype(int)
            self.feature_map[feature_name].append(dummy_name)
            self.write_sparse_column(dummy_name)
        self.drop(feature_name)

    def continuous(self, feature_name, squared=False, cubed=False, log=False, topcode=False, missing=True):
        """
        Center continuous values at mean=0 and standardize to sd=1.
        Create missing indicator and mean-impute missing values.
        Optionally add higher-order terms (before standardization).
        """
        assert feature_name in self.outstanding_features, feature_name
        # Create missing indicator
        missing_mask = self.X[feature_name].isna()
        missing_training_mask = (missing_mask & self.training_mask)
        if missing and missing_training_mask.any():
            dummy_name = sanitize_name(f"{feature_name}_MISSING")
            assert dummy_name not in self.X.columns, dummy_name
            self.X[dummy_name] = missing_mask.astype(int)
            self.feature_map[feature_name].append(dummy_name)
            self.write_sparse_column(dummy_name)
        # Top code
        if topcode:
            assert 0 < topcode and topcode < 1
            t = np.quantile(self.X.loc[self.training_mask, feature_name], topcode)
            self.X.loc[self.X[feature_name] >= t, feature_name] = t
        # Log transform
        if log:
            self.X.loc[:, feature_name] = np.log(self.X[feature_name])
        # Sanitize name
        new_name = sanitize_name(feature_name)
        assert new_name not in self.X.columns, new_name
        self.X[new_name] = self.X[feature_name]
        names = [new_name]
        # Add squared/cubed terms
        if squared:
            squared_name = sanitize_name(f"{feature_name}_SQAURED")
            assert squared_name not in self.X.columns, squared_name
            self.X[squared_name] = self.X[feature_name] * self.X[feature_name]
            names.append(squared_name)
        if cubed:
            cubed_name = sanitize_name(f"{feature_name}_CUBED")
            assert cubed_name not in self.X.columns, cubed_name
            self.X[cubed_name] = self.X[feature_name] * self.X[feature_name] * self.X[feature_name]
            names.append(cubed_name)
        # Standardize features
        for name in names:
            mean  = self.X.loc[self.training_mask, name].mean()
            stdev = self.X.loc[self.training_mask, name].std()
            if stdev != 0:
                self.X.loc[:, name] = self.X[name].subtract(mean).divide(stdev)
            else:
                print(f"WARNING: {name} has 0 stdev")
            self.X[name].fillna(0, inplace=True)
            self.feature_map[feature_name].append(name)
            self.write_sparse_column(name)
        self.drop(feature_name)

    def hurdle(self, feature_name, value=0, squared=False, cubed=False, log=False, topcode=False):
        """
        Expand a "hurdle" variable into into extensive and intensive
        margins. The extensive margin becomes a dummy variable and the
        the intensive margin a continuous feature.
        """
        assert feature_name in self.outstanding_features, feature_name
        # Create missing indicator
        missing_mask = self.X[feature_name].isna()
        missing_training_mask = (missing_mask & self.training_mask)
        if missing_training_mask.any():
            dummy_name = sanitize_name(f"{feature_name}_MISSING")
            assert dummy_name not in self.X.columns, dummy_name
            self.X[dummy_name] = missing_mask.astype(int)
            self.feature_map[feature_name].append(dummy_name)
            self.write_sparse_column(dummy_name)
        # Create non-zero indicator (intensive margin)
        hurdle_mask = (self.X[feature_name] <= value)
        if hurdle_mask.loc[self.training_mask].any():  
            dummy_name = sanitize_name(f"{feature_name}_NONZERO")
            assert dummy_name not in self.X.columns, dummy_name
            self.X[dummy_name] = (self.X[feature_name] > value).astype(int)
            self.feature_map[feature_name].append(dummy_name)
            self.write_sparse_column(dummy_name)
        # Create extensive margin
        self.X.loc[hurdle_mask, feature_name] = np.NaN
        self.continuous(feature_name, squared=squared, cubed=cubed, log=log, topcode=topcode, missing=False)


    def interactions(self):
        """
        Use the feature map to determine all pairwise interactions between
        engineered features.
        """
        names = list(self.feature_map.keys())
        for i, name1 in enumerate(names):
            features1 = self.feature_map[name1]
            for name2 in names[i+1:]:
                features2 = self.feature_map[name2]
                for feature1 in features1:
                    for feature2 in features2:
                        self.interaction(feature1, feature2)

    def interaction(self, feature1, feature2):
        """
        Add a pairwise interaction between feature1 and feature2.
        """
        assert feature1 in self.X.columns, feature1
        assert feature2 in self.X.columns, feature2
        interaction_name = f"{feature1}_X_{feature2}"
        interaction = self.X[feature1] * self.X[feature2]
        if len(interaction.loc[self.training_mask].unique()) == 1:
            print("WARNING: dropping empty interaction", interaction_name)
        elif interaction.equals(self.X[feature1]) or interaction.equals(self.X[feature2]):
            print("WARNING: dropping redundant interaction", interaction_name) 
        else:
            self.write_sparse_column(interaction_name, interaction)

    def close_csc(self):
        """
        Write out the final line and close the output file.
        """
        if self.outstanding_features:
            print("WARNING: outstanding features that have not been processed:")
            print("\n".join(self.outstanding_features))
        self.csc_file.write("#csc end ncol={}\n".format(len(self.csc_columns)))
        self.csc_file.close()

    def to_csv(self, columns, filename):
        """
        Write the specified columns of X to a csv file.
        """
        self.X[columns].to_csv(filename, index=False)


def sanitize_name(feature_name):
    """
    Remove non-alphanumeric characters and replace with underscores.
    """
    return re.sub(r"[^A-Za-z0-9]", "_", feature_name).upper()


if __name__ == "__main__":
    main()
