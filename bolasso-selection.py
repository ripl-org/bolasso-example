import pandas as pd
import sys

# Parse command-line arguments
threshold  = float(sys.argv[1])
coef_files = sys.argv[2:-3]
avg_file   = sys.argv[-3]
freq_file  = sys.argv[-2]
out_file   = sys.argv[-1]

# Load coefficients and calculate non-zero frequency
var = pd.read_csv(coef_files[0], index_col="var")
var["freq"] = (var.coef != 0).astype(int)
for coef_file in coef_files[1:]:
    var1 = pd.read_csv(coef_file, index_col="var")
    var["coef"] += var1.coef
    var["freq"] += (var1.coef != 0).astype(int)
var["coef"] /= len(coef_files)
var["freq"] /= len(coef_files)

# Write average coefficients to file
var[var.coef != 0].to_csv(avg_file)

# Write frequencies to file
del var["coef"]
var.drop("intercept", inplace=True)
var.to_csv(freq_file)

# Select variables at >= threshold
selected = set(var[var.freq >= threshold].index)

# Ensure that base variables are preserved for any selected interaction
for name in list(selected):
    if "_X_" in name:
        var1, _, var2 = name.partition("_X_")
        selected.add(var1)
        selected.add(var2)

# Write selected variable list
with open(out_file, "w") as f:
    print("var", file=f)
    print("\n".join(sorted(selected)), file=f)
