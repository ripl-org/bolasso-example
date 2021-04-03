library(assertthat)
library(AUC)
library(Matrix)

args <- commandArgs(trailingOnly=TRUE)

set.seed(args[1])

matrix_file   <- args[2]
selected_file <- args[3]
out_file      <- args[4]
pred_file     <- args[5]

load(matrix_file, verbose=TRUE)
y_train <- y[train]

selected <- read.csv(selected_file, stringsAsFactors=FALSE)$var
print(selected)

X_train <- X_train[,selected]

k <- kappa(X_train, exact=TRUE)
print(paste0("condition number (kappa): ", k))
assert_that(k < 100)

# add intercept
X_train <- cbind(X_train, 1)

model <- glm.fit(x=X_train, y=y_train, family=binomial())
params <- summary.glm(model)$coefficients

# Convert coefficients to odds ratios and standard errors to 95% C.I.
odds <- exp(params[,1])
ci_lower <- exp(params[,1] - 1.96*params[,2])
ci_upper <- exp(params[,1] + 1.96*params[,2])
write.csv(data.frame(var=c(selected, "intercept"), odds=odds, ci_lower=ci_lower, ci_upper=ci_upper, p=params[,4]),
          file=out_file, row.names=FALSE)

# Predict on test data with OLS
X_test <- cbind(X_test[,selected], 1)
coef <- as.matrix(model$coef)
eta <- as.matrix(X_test) %*% as.matrix(coef)
y_test <- y[test]
y_pred <- exp(eta)/(1 + exp(eta))
print(paste0("AUC: ", auc(roc(y_pred, as.factor(y_test)))))
write.csv(data.frame(y_pred=y_pred, y_test=y_test), file=pred_file, row.names=FALSE)