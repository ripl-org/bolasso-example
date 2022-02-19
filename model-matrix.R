library(assertthat)
library(data.table)
library(Matrix)
library(tidyverse)

args <- commandArgs(trailingOnly=TRUE)
n <- length(args)

features_file <- args[1]
csc_file      <- args[2]
corr_file     <- args[3]
matrix_file   <- args[4]

cat("Loading features file\n")

features <- fread(features_file)

cat("Creating training/testing flags\n")

train <- features[,1] == "TRAIN"
cat("Found", sum(train), "training observations\n")

test <- features[,1] == "TEST"
cat("Found", sum(test), "testing observations\n")

cat("Extracting outcome vector\n")

y <- features[,2]

cat("Extracting weights vector\n")

weights <- features[,3]

cat("Loading compressed sparse columns\n")

csc <- gzfile(csc_file, "rt")

# Read header line
line <- scan(file=csc, nlines=1, what=character(), quiet=TRUE)
assert_that(line[1] == "#csc")
assert_that(line[2] == "start")
N <- strtoi(gsub("nrow=", "", line[3])) # number of rows

M <- 0 # number of columns
names <- list()
i <- list()
j <- list()
x <- list()

# Read column lines
line <- scan(file=csc, nlines=1, what=character(), quiet=TRUE)
while (line[1] != "#csc") {
    assert_that(line[1] == "column")
    M <- M + 1
    names[[M]] <- line[2]
    i[[M]] <- scan(file=csc, nlines=1, what=integer(), quiet=TRUE) + 1 # Adjust for 1-based indexing
    j[[M]] <- rep(M, length(i[[M]]))
    x[[M]] <- scan(file=csc, nlines=1, what=numeric(), quiet=TRUE)
    line <- scan(file=csc, nlines=1, what=character(), quiet=TRUE)
}

# Read ending line
assert_that(line[1] == "#csc")
assert_that(line[2] == "end")
assert_that(M == strtoi(gsub("ncol=", "", line[3])))

close(csc)

cat("Loaded", N, "rows X", M, "columns\n")

cat("Creating sparse model matrix\n")

i <- unlist(i)
gc()
j <- unlist(j)
gc()
x <- unlist(x)
gc()

X <- sparseMatrix(i=i, j=j, x=x)
colnames(X) <- names
rm("i", "j", "x")
gc()

X_train <- X[which(train),]
X_test  <- X[which(test), ]
rm("X")
gc()

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
