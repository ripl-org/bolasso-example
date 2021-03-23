library(gamlr)

# Parse command-line arguments
args <- commandArgs(trailingOnly=TRUE)
matrix_file <- args[1]
random_seed <- strtoi(args[2])
bootstrap   <- strtoi(args[3])
coef_file   <- args[4]
pred_file   <- args[5]

# Load R objects X (sparse or dense matrix) and Y (vector)
load(matrix_file)
y_train <- y[train]

# Set random seed for reproducibility
set.seed(random_seed)

# Re-seed using the bootstrap number, then generate an index vector
# for sampling a bootstrap replicate
set.seed(sample.int(2147483647, bootstrap)[bootstrap])
idx <- sample(1:nrow(X_train), nrow(X_train), replace=TRUE)

# Run LASSO with cross-validation
model <- cv.gamlr(x=X_train[idx,],
                  y=y_train[idx],
                  family="binomial",
                  standardize=FALSE)

# Write out coefficients
coefs <- as.matrix(coef(model, select="1se"))
write.csv(
    data.frame(var=rownames(coefs), coef=coefs[,1]),
    row.names=FALSE,
    file=coef_file
)

# Write out predictions on test data
predicted <- as.vector(predict(model, newdata=X_test, select="1se"))
write.csv(
    data.frame(predicted=exp(predicted), actual=y[test]),
    row.names=FALSE,
    file=pred_file
)