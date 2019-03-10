#' unroll representation of RNN running on non CUDA device with attention
#' 
#' @export
#' unroll representation of RNN running on non CUDA device
#' 
#' @param config Either seq-to-one or one-to-one
#' @param cell_type Type of RNN cell: either gru or lstm
#' @param num_rnn_layer int, number of stacked layers
#' @param seq_len int, number of time steps to unroll
#' @param num_hidden int, size of the state in each RNN layer
#' @param num_embed  int, default = NULL - no embedding. Dimension of the embedding vectors
#' @param num_decode int, number of output variables in the decoding layer
#' @param input_size int, number of levels in the data - only used for embedding
#' @param dropout 
#' 
#' @export
rnn.graph.unroll.encode <- function(num_bi,
                                    num_uni,
                                    seq_len, 
                                    input_size = NULL,
                                    num_embed = NULL, 
                                    num_hidden,
                                    num_decode,
                                    dropout = 0,
                                    ignore_label = -1,
                                    loss_output = NULL, 
                                    init.state = NULL,
                                    cell_type = "lstm", 
                                    masking = F, 
                                    output_last_state = F,
                                    prefix = "",
                                    data_name = "data",
                                    bidirectional = F,
                                    residual = T,
                                    reverse_input = F) {
  
  if (!is.null(num_embed)) embed.weight <- mx.symbol.Variable(paste0(prefix, "embed.weight"))
  
  cls.weight <- mx.symbol.Variable(paste0(prefix, "cls.weight"))
  cls.bias <- mx.symbol.Variable(paste0(prefix, "cls.bias"))
  
  rnn_bi_weight <- lapply(1:num_bi, function(i) {
    mx.symbol.Variable(paste0(prefix, "bi_", i, "_weight"))
  })
  rnn_uni_weight <- lapply(1:num_uni, function(i) {
    mx.symbol.Variable(paste0(prefix, "uni_", i, "_weight"))
  })
  
  # embeding layer
  data <- mx.symbol.Variable(data_name)
  
  if (masking) {
    seq.mask <- mxnet:::mx.varg.symbol.internal.not_equal_scalar(alist = list(data=data, scalar = 0))
    seq.mask <- mx.symbol.sum_axis(data = seq.mask, axis = 1)
  }
  
  data = mx.symbol.swapaxes(data = data, dim1 = 0, dim2 = 1, name = paste0(prefix, "swap_pre"))
  
  if (!is.null(num_embed)) {
    data <- mx.symbol.Embedding(data = data, input_dim = input_size,
                                weight=embed.weight, output_dim = num_embed, name = paste0(prefix, "embed"))
  }
  
  
  # front 
  for (i in 1:num_bi) {
    front <- mx.symbol.RNN(data = data, parameters = rnn_bi_weight[[i]], state_size = num_hidden, num_layers = 1, 
                             mode = "lstm", p=dropout, bidirectional=F, name = paste0(prefix, "RNN_bi_", i))
  }
  
  # reverse 
  data_rev <- mx.symbol.SequenceReverse(data = data, use.sequence.length = T, sequence_length = seq.mask, name = paste0(prefix, "reverse_pre"))
  for (i in 1:num_bi) {
    data_rev <- mx.symbol.RNN(data = data_rev, parameters = rnn_bi_weight[[i]], state_size = num_hidden, num_layers = 1, 
                             mode = "lstm", p=dropout, bidirectional=F, name = paste0(prefix, "RNN_bi_rev_", i))
  }
  data_rev <- mx.symbol.SequenceReverse(data = data_rev, use.sequence.length = T, sequence_length = seq.mask, name = paste0(prefix, "reverse_post"))
  
  # data <- front + data_rev
  data <- mx.symbol.concat(data = c(front, data_rev), num.args = 2, dim = 2)
  # data <- mx.symbol.FullyConnected(data = data, flatten = F, num_hidden = num_hidden)
  
  for (i in 1:num_uni) {
    rnn <- mx.symbol.RNN(data = data, parameters = rnn_uni_weight[[i]], state_size = num_hidden, num_layers = 1, 
                         mode = "lstm", p=dropout, bidirectional=F, name = paste0(prefix, "RNN_uni_", i))
    if(i>1) data <- data + rnn else data <- rnn
    # data <- data + rnn
  }
  
  # concat <- mx.symbol.cast(concat, dtype = "float32")
  if (masking) {
    mask <- mx.symbol.SequenceMask(data = data, use.sequence.length = T, sequence_length = seq.mask, value = 0, name = paste0(prefix, "mask"))
  } else mask <- mx.symbol.identity(data = data, name = paste0(prefix, "mask"))
  
  value <- mx.symbol.swapaxes(data = mask, dim1 = 0, dim2 = 1, name = paste0(prefix, "value"))
  
  return(list(value = value, length = seq.mask))
  
}
