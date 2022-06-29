import os
from scons_remote.environment_remote import EnvironmentRemote

env = EnvironmentRemote(ENV=os.environ)

N_BOOTSTRAP = 100
RANDOM_SEED = 9477179
BOLASSO_THRESHOLD = 0.9

# The SCons cache stores previously calculated results, and the cache directory can
# be shared among collaborators to accelerate calculations, since the first person to
# build caches the results for others to retrieve.
env.CacheDir("cache")

# NOTE: This is set up to run in the NJ Dev AWS Account
client_args = {
    'region_name': 'us-west-2'
}

instance_args = {
    'ImageId': 'ami-0030721ee0ca43dfe',
    'InstanceType': 't2.2xlarge',
    'KeyName': 'sandbox',
    'MaxCount': 1,
    'MinCount': 1,
    'SecurityGroupIds': ['sg-05034d98ec6368ee7'],
    'InstanceInitiatedShutdownBehavior': 'terminate'
}

ssh_args = {
    'user': 'ubuntu',
    'connect_kwargs': {
        'key_filename': 'path/to/sandbox.pem'
    }
}

env.connection_initialize(client_args, instance_args, ssh_args)

# Feature engineering
env.CommandRemote(
    target="scratch/features.csv",
    source=[
        "feature-engineering.py",
        "data/uci-adult-train.csv",
        "data/uci-adult-test.csv"
    ],
    action=env.ActionRemote(cmd="python3")
)

# Model matrix
env.CommandRemote(
    target=[
        "scratch/top-correlations.csv",
        "scratch/model-matrix.Rdata"
    ],
    source=[
        "model-matrix.R",
        Value("salary_50k"),
        Value("subset"),
        Value("fnlwgt"),
        "scratch/features.csv"
    ],
    action=env.ActionRemote(cmd="Rscript")
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
        "scratch/bolasso-ensemble-coefficients.csv",
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
env.CommandRemote(
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
    action=env.ActionRemote(cmd="Rscript")
)

# Codebooks
env.Command(
    target=[
        "scratch/train_codebook.html"
    ],
    source=[
        "data/uci-adult-train.csv"
    ],
    action="codebooks --na_values ? --output $TARGET $SOURCE"
)
env.Command(
    target=[
        "scratch/features_codebook.html"
    ],
    source=[
        "scratch/features.csv"
    ],
    action="codebooks --output $TARGET $SOURCE"
)
