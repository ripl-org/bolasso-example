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
