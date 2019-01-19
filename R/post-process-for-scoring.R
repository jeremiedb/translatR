# post-process of translation
library(readr)

data <- read_lines(file = "data/trans_test_wmt_en_fr_rnn_72_Capital.txt")

data <- gsub(pattern = " l ", replacement = " l'", x = data)
data <- gsub(pattern = " d ", replacement = " d'", x = data)
data <- paste0(data, ".", sep = "")

write_lines(data, path = "data/trans_test_wmt_en_fr_rnn_72_Capital.txt")
