library(assertthat)
library(data.table)
library(Matrix)
library(tidyverse)

args <- commandArgs(trailingOnly=TRUE)
n <- length(args)

outcome_name  <- args[1]
subset_name   <- args[2]
weights_name  <- args[3]
feature_file  <- args[4]
corr_file     <- args[5]
matrix_file   <- args[6]

cat("Loading features file\n")

features <- fread(feature_file)

cat("Loaded", nrow(features), "rows X", ncol(features), "columns\n")

cat("Creating training/testing sets\n")

train <- features[,get(subset_name)] == "TRAIN"
cat("Found", sum(train), "training observations\n")

test <- features[,get(subset_name)] == "TEST"
cat("Found", sum(test), "testing observations\n")

features <- features[,(subset_name):=NULL] # Remove subset column

cat("Validating features\n")
constant = c()
for (name in colnames(features)) {
    x <- features[,get(name)]
    if (length(unique(x)) == 1) {
        constant = c(constant, name)
    } else {
        type <- class(x)
        assert_that(
            type == "numeric" || type == "integer",
            msg=paste(name, "has type", type, "instead of numeric/integer")
        )
        assert_that(!any(is.na(x)), msg=paste(name, "has NA values"))
    }
}

cat("Dropping", nrow(constant), "features that are constant in training data\n")

for (name in constant) {
    features <- features[,(name):=NULL]
}

cat("Extracting outcome vector\n")

y <- features[,get(outcome_name)]
features <- features[,(outcome_name):=NULL] # Remove outcomes column

cat("Extracting weights vector\n")

weights <- features[,get(weights_name)]
features <- features[,(weights_name):=NULL] # Remove weights column

cat("Creating sparse model matrix\n")

X_train <- Matrix(as.matrix(features[train,]), sparse=TRUE)
X_test <- Matrix(as.matrix(features[test,]), sparse=TRUE)

cat("Calculating top pairwise correlations for training features\n")

# https://stackoverflow.com/questions/5888287/running-cor-or-any-variant-over-a-sparse-matrix-in-r
sparse.cor <- function(x) {
    n <- nrow(x)

    cMeans <- colMeans(x)
    cSums <- colSums(x)

    # Calculate the population covariance matrix.
    # There's no need to divide by (n-1) as the std. dev is also calculated the same way.
    # The code is optimized to minize use of memory and expensive operations
    covmat <- tcrossprod(cMeans, (-2*cSums+n*cMeans))
    crossp <- as.matrix(crossprod(x))
    covmat <- covmat+crossp

    sdvec <- sqrt(diag(covmat)) # standard deviations of columns
    covmat/crossprod(t(sdvec)) # correlation matrix
}
cor <- sparse.cor(X_train)
# http://r.789695.n4.nabble.com/return-only-pairwise-correlations-greater-than-given-value-td4079028.html
cor[upper.tri(cor, TRUE)] <- NA
i <- which(abs(cor) >= 0.99, arr.ind=TRUE)
drop <- data.frame(matrix(colnames(cor)[as.vector(i)], ncol=2), value=cor[i])
drop$col = ifelse(str_detect(drop$X1, "_X_"), drop$X1, drop$X2)

cat("Dropping", nrow(drop), "highly correlated features\n")

i <- which(colnames(X_train) %in% drop$col)
X_train <- X_train[,-i]
i <- which(colnames(X_test) %in% drop$col)
X_test <- X_test[,-i]

cat("Writing out top correlated features\n")

cor <- sparse.cor(X_train)
cor[upper.tri(cor, TRUE)] <- NA
i <- which(abs(cor) >= 0.7, arr.ind=TRUE)
top <- data.frame(matrix(colnames(cor)[as.vector(i)], ncol=2), value=cor[i])
write.csv(top, file=corr_file)

cat("Saving\n")

save(X_train, X_test, y, weights, train, test, file=matrix_file)
