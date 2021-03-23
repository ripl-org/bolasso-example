import os

env = Environment(ENV=os.environ)

env.Command(
    target="scratch/features.csv",
    source=[
        "feature-engineering.py",
        "data/uci-adult-train.csv",
        "data/uci-adult-test.csv"
    ],
    action="python $SOURCES $TARGETS"
)

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