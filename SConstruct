import os

env = Environment(ENV=os.environ)

N_BOOTSTRAP = 100
RANDOM_SEED = 9477179
BOLASSO_THRESHOLD = 0.9

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

# BOLASSO replicates
for i in range(1, N_BOOTSTRAP+1):
    env.Command(
        target=[
            f"scratch/bolasso/coefficients.{i}.csv",
            f"scratch/bolasso/predictions.{i}.csv"
        ],
        source=[
            "bolasso-replicate.R",
            "scratch/model-matrix.RData",
            Value(RANDOM_SEED),
            Value(i)
        ],
        action="Rscript $SOURCES $TARGETS"
    )

# BOLASSO selection
env.Command(
    target=[
        "scratch/bolasso-frequencies.csv",
        "scratch/bolasso-selection.csv"
    ],
    source=[
        "bolasso-selection.py",
        Value(BOLASSO_THRESHOLD)
    ] + [f"scratch/bolasso/coefficients.{i}.csv" for i in range(1, N_BOOTSTRAP+1)],
    action="python $SOURCES $TARGETS"
)

# BOLASSO ensemble
env.Command(
    target=[
        "scratch/bolasso-ensemble-predictions.csv"
    ],
    source=[
        "bolasso-ensemble.R",
        "scratch/model-matrix.RData"
    ] + [f"scratch/bolasso/predictions.{i}.csv" for i in range(1, N_BOOTSTRAP+1)],
    action="Rscript $SOURCES $TARGETS"
)

# Post-LASSO
env.Command(
    target=[
        "scratch/post-lasso-coefficients.csv",
        "scratch/post-lasso-predictions.csv"
    ],
    source=[
        "post-lasso.R",
        Value(RANDOM_SEED),
        "scratch/model-matrix.RData",
        "scratch/bolasso-selection.csv"
    ],
    action="Rscript $SOURCES $TARGETS"
)
