suppressMessages({library(scrutiny); library(tibble)})

args <- commandArgs(trailingOnly=TRUE)
infile  <- args[1]
outfile <- args[2]

d <- read.csv(infile, colClasses=c(mean_str="character"), stringsAsFactors=FALSE)
cat("rows read:", nrow(d), "\n")

# scrutiny grim_map: x = mean as string (decimals preserved), n = sample size.
# Default rounding = up_or_down (canonical GRIM). consistency==TRUE -> GRIM-consistent.
gm_in <- tibble(x = d$mean_str, n = as.integer(d$n))

# grim_map vectorized over the tibble
res <- grim_map(gm_in)
# res has columns x, n, consistency (logical). Align by row order (grim_map preserves order).
stopifnot(nrow(res) == nrow(d))

d$grimmer_consistent <- as.integer(res$consistency)        # 1 = consistent (NOT flagged)
d$grimmer_inconsistent <- as.integer(!res$consistency)     # 1 = inconsistent (flagged)
# testable: grim_map sets consistency=TRUE when not testable? Track NA just in case.
d$grimmer_na <- as.integer(is.na(res$consistency))

write.csv(d, outfile, row.names=FALSE)
cat("wrote:", outfile, "\n")
cat("grimmer_inconsistent:", sum(d$grimmer_inconsistent, na.rm=TRUE), "\n")
cat("grimmer NA:", sum(d$grimmer_na), "\n")
