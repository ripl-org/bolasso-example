library(AUC)
library(Matrix)

args <- commandArgs(trailingOnly=TRUE)

matrix_file <- args[1]
pred_file   <- args[2]
pred_files  <- args[3:length(args)-1]
out_file    <- args[length(args)]

load(matrix_file)

# Average predictions across BOLASSO replicates
y_pred <- read.csv(pred_file)$predicted
for (pred_file in pred_files) {
    y_pred <- y_pred + read.csv(pred_file)$predicted
}
y_pred <- y_pred / (length(pred_files) + 1)

y_test <- y[test]

print(paste0("AUC: ", auc(roc(y_pred, as.factor(y_test)))))
write.csv(data.frame(y_pred=y_pred, y_test=y_test), file=out_file, row.names=FALSE)