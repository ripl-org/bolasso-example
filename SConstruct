import os

env = Environment(ENV=os.environ)
N_BOOTSTRAP = 100
RANDOM_SEED = 9477179

# Feature engineering
env.Command(
    target="scratch/features.csv",
    source=[
        "feature-engineering.py",
        "data/uci-adult-train.csv",
        "data/uci-adult-test.csv"
    ],
    action="python $SOURCES $TARGETS"
)

# Model matrix
env.Command(
    target=[
        "scratch/top-correlations.csv",
        "scratch/model-matrix.Rdata"
    ],
    source=[
        "model-matrix.R",
        Value("salary_50k"),
        Value("subset"),
        "scratch/features.csv"
    ],
    action="Rscript $SOURCES $TARGETS"
)

# BOLASSO
for i in range(1, N_BOOTSTRAP+1):
    env.Command(
        target=[
            "scratch/bolasso/coefficients.{}.csv".format(i),
            "scratch/bolasso/predictions.{}.csv".format(i)
        ],
        source=[
            "bolasso-replicate.R",
            "scratch/model-matrix.RData",
            Value(RANDOM_SEED),
            Value(i)
        ],
        action="Rscript $SOURCES $TARGETS"
    )
