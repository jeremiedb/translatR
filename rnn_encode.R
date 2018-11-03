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
rnn.graph.unroll.encode <- function(num_rnn_layer, 
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
                             reverse_input = F) {
  
  if (!is.null(num_embed)) embed.weight <- mx.symbol.Variable(paste0(prefix, "embed.weight"))
  
  # Initial state
  if (is.null(init.state) & output_last_state) {
    init.state <- lapply(1:num_rnn_layer, function(i) {
      if (cell_type=="lstm") {
        state <- list(h = mx.symbol.Variable(paste0("init_", prefix, i, "_h")),
                      c = mx.symbol.Variable(paste0("init_", prefix, i, "_c")))
      } else if (cell_type=="gru") {
        state <- list(h = mx.symbol.Variable(paste0("init_", prefix, i, "_h")))
      }
      return (state)
    })
  }
  
  cls.weight <- mx.symbol.Variable(paste0(prefix, "cls.weight"))
  cls.bias <- mx.symbol.Variable(paste0(prefix, "cls.bias"))
  
  param.cells <- lapply(1:num_rnn_layer, function(i) {
    
    if (cell_type=="lstm") {
      cell <- list(i2h.weight = mx.symbol.Variable(paste0(prefix, "l", i, ".i2h.weight")),
                   i2h.bias = mx.symbol.Variable(paste0(prefix, "l", i, ".i2h.bias")),
                   h2h.weight = mx.symbol.Variable(paste0(prefix, "l", i, ".h2h.weight")),
                   h2h.bias = mx.symbol.Variable(paste0(prefix, "l", i, ".h2h.bias")))
    } else if (cell_type=="gru") {
      cell <- list(gates.i2h.weight = mx.symbol.Variable(paste0(prefix, "l", i, ".gates.i2h.weight")),
                   gates.i2h.bias = mx.symbol.Variable(paste0(prefix, "l", i, ".gates.i2h.bias")),
                   gates.h2h.weight = mx.symbol.Variable(paste0(prefix, "l", i, ".gates.h2h.weight")),
                   gates.h2h.bias = mx.symbol.Variable(paste0(prefix, "l", i, ".gates.h2h.bias")),
                   trans.i2h.weight = mx.symbol.Variable(paste0(prefix, "l", i, ".trans.i2h.weight")),
                   trans.i2h.bias = mx.symbol.Variable(paste0(prefix, "l", i, ".trans.i2h.bias")),
                   trans.h2h.weight = mx.symbol.Variable(paste0(prefix, "l", i, ".trans.h2h.weight")),
                   trans.h2h.bias = mx.symbol.Variable(paste0(prefix, "l", i, ".trans.h2h.bias")))
    } else if (cell_type=="straight") {
      cell <- list(input.weight = mx.symbol.Variable(paste0(prefix, "l", i, "input.weight")),
                   input.bias = mx.symbol.Variable(paste0(prefix, "l", i, "input.bias")),
                   write.weight = mx.symbol.Variable(paste0(prefix, "l", i, "write.weight")),
                   write.bias = mx.symbol.Variable(paste0(prefix, "l", i, "write.bias")),
                   mem.weight = mx.symbol.Variable(paste0(prefix, "l", i, "mem.weight")),
                   mem.bias = mx.symbol.Variable(paste0(prefix, "l", i, "mem.bias")),
                   highway.weight = mx.symbol.Variable(paste0(prefix, "l", i, "highway.weight")),
                   highway.bias = mx.symbol.Variable(paste0(prefix, "l", i, "highway.bias")),
                   read.weight = mx.symbol.Variable(paste0(prefix, "l", i, "read.weight")),
                   read.bias = mx.symbol.Variable(paste0(prefix, "l", i, "read.bias")))
    } else if (cell_type=="rich") {
      cell <- list(input.weight = mx.symbol.Variable(paste0(prefix, "l", i, "input.weight")),
                   input.bias = mx.symbol.Variable(paste0(prefix, "l", i, "input.bias")),
                   mem.weight = mx.symbol.Variable(paste0(prefix, "l", i, "mem.weight")),
                   mem.bias = mx.symbol.Variable(paste0(prefix, "l", i, "mem.bias")),
                   write.in.weight = mx.symbol.Variable(paste0(prefix, "l", i, "write.in.weight")),
                   write.in.bias = mx.symbol.Variable(paste0(prefix, "l", i, "write.in.bias")),
                   write.c.weight = mx.symbol.Variable(paste0(prefix, "l", i, "write.c.weight")),
                   write.c.bias = mx.symbol.Variable(paste0(prefix, "l", i, "write.c.bias")),
                   highway.weight = mx.symbol.Variable(paste0(prefix, "l", i, "highway.weight")),
                   highway.bias = mx.symbol.Variable(paste0(prefix, "l", i, "highway.bias")),
                   read.in.weight = mx.symbol.Variable(paste0(prefix, "l", i, "read.in.weight")),
                   read.in.bias = mx.symbol.Variable(paste0(prefix, "l", i, "read.in.bias")),
                   read.c.weight = mx.symbol.Variable(paste0(prefix, "l", i, "read.c.weight")),
                   read.c.bias = mx.symbol.Variable(paste0(prefix, "l", i, "read.c.bias")))
    }
    return (cell)
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
  
  # Reverse data
  if (reverse_input) {
    if (masking) {
      data <- mx.symbol.SequenceReverse(data = data, use.sequence.length = T, sequence_length = seq.mask, name = paste0(prefix, "reverse"))  
    } else {
      data <- mx.symbol.SequenceReverse(data = data, use.sequence.length = F, name = paste0(prefix, "reverse"))
    }
  } else if (bidirectional) {
    if (masking) {
      data_rev <- mx.symbol.SequenceReverse(data = data, use.sequence.length = T, sequence_length = seq.mask, name = paste0(prefix, "reverse"))  
    } else {
      data_rev <- mx.symbol.SequenceReverse(data = data, use.sequence.length = F, name = paste0(prefix, "reverse"))
    }
  } 
  
  
  data <- mx.symbol.split(data = data, axis = 0, num.outputs = seq_len, squeeze_axis = T)
  if (bidirectional) data_rev <- mx.symbol.split(data = data_rev, axis = 0, num.outputs = seq_len, squeeze_axis = T)
  
  last.hidden <- list()
  last.states <- list()
  
  for (seqidx in 1:seq_len) {
    hidden <- data[[seqidx]]
    
    for (i in 1:num_rnn_layer) {
      
      if (seqidx==1) prev.state <- init.state[[i]] else 
        prev.state <- last.states[[i]]
      
      if (cell_type=="lstm") {
        cell.symbol <- lstm.cell
      } else if (cell_type=="gru"){
        cell.symbol <- gru.cell
      } else if (cell_type=="straight"){
        cell.symbol <- straight.cell
      } else if (cell_type=="rich"){
        cell.symbol <- rich.cell
      }
      
      next.state <- cell.symbol(num_hidden = num_hidden, 
                                indata = hidden,
                                prev.state = prev.state,
                                param = param.cells[[i]],
                                seqidx = seqidx, 
                                layeridx = i,
                                dropout = dropout,
                                prefix = prefix)
      
      hidden <- next.state$h
      last.states[[i]] <- next.state
    }
    
    # Aggregate outputs from each timestep
    last.hidden <- c(last.hidden, hidden)
  }
  
  
  # RNN on reverse sequence
  if (bidirectional) {
    
    last.hidden.rev <- list()
    last.states.rev <- list()
    
    for (seqidx in 1:seq_len) {
      hidden <- data_rev[[seqidx]]
      
      for (i in 1:num_rnn_layer) {
        
        if (seqidx==1) prev.state <- init.state[[i]] else 
          prev.state <- last.states.rev[[i]]
        
        if (cell_type=="lstm") {
          cell.symbol <- lstm.cell
        } else if (cell_type=="gru"){
          cell.symbol <- gru.cell
        } else if (cell_type=="straight"){
          cell.symbol <- straight.cell
        } else if (cell_type=="rich"){
          cell.symbol <- rich.cell
        }
        
        next.state <- cell.symbol(num_hidden = num_hidden, 
                                  indata = hidden,
                                  prev.state = prev.state,
                                  param = param.cells[[i]],
                                  seqidx = seqidx, 
                                  layeridx = i,
                                  dropout = dropout,
                                  prefix = paste0(prefix, "rev_"))
        
        hidden <- next.state$h
        last.states.rev[[i]] <- next.state
      }
      
      # Aggregate outputs from each timestep
      last.hidden.rev <- c(last.hidden.rev, hidden)
    }
  }
  ### End of bidirectionnal RNN ###
  
  if (output_last_state) {
    out.states = mx.symbol.Group(unlist(last.states))
  }
  
  # concat hidden units - concat seq_len blocks of dimension num_hidden x batch.size
  # concat <- mx.symbol.stack(data = last.hidden, axis = 0, num_args = seq_len, name = paste0(prefix, "concat"))
  concat <- mx.symbol.concat(data = last.hidden, num.args = seq_len, dim = 0, name = paste0(prefix, "concat"))
  concat <- mx.symbol.reshape(data = concat, shape = c(num_hidden, -1, seq_len), name = paste0(prefix, "rnn_reshape"))
  
  if (reverse_input) {
    if (masking) {
      concat <- mx.symbol.SequenceReverse(data = concat, use.sequence.length = T, sequence_length = seq.mask, name = paste0(prefix, "reverse_post"))  
    } else {
      concat <- mx.symbol.SequenceReverse(data = concat, use.sequence.length = F, name = paste0(prefix, "reverse_post"))
    }
  } else if (bidirectional) {
    concat_rev <- mx.symbol.concat(data = last.hidden.rev, num.args = seq_len, dim = 0, name = paste0(prefix, "concat_rev"))
    concat_rev <- mx.symbol.reshape(data = concat_rev, shape = c(num_hidden, -1, seq_len), name = paste0(prefix, "rnn_reshape_rev"))
    if (masking) {
      concat_rev <- mx.symbol.SequenceReverse(data = concat_rev, use.sequence.length = T, sequence_length = seq.mask, name = paste0(prefix, "reverse_post"))  
    } else {
      concat_rev <- mx.symbol.SequenceReverse(data = concat_rev, use.sequence.length = F, name = paste0(prefix, "reverse_post"))
    }
    concat <- concat + concat_rev
  }
  
  if (masking) {
    mask <- mx.symbol.SequenceMask(data = concat, use.sequence.length = T, sequence_length = seq.mask, value = 0, name = paste0(prefix, "mask"))
  } else mask <- mx.symbol.identity(data = concat, name = paste0(prefix, "mask"))
  
  value <- mx.symbol.swapaxes(data = mask, dim1 = 0, dim2 = 1, name = paste0(prefix, "value"))
  
  if (output_last_state) {
    return(mx.symbol.Group(c(value, out.states)))
  } else return(value)
  
}
