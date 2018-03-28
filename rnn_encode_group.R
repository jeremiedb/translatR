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
                             config,
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
    } else if (cell_type=="sru") {
      cell <- list(input.weight = mx.symbol.Variable(paste0(prefix, "l", i, "input.weight")),
                   input.bias = mx.symbol.Variable(paste0(prefix, "l", i, "input.bias")),
                   forget.weight = mx.symbol.Variable(paste0(prefix, "l", i, "forget.weight")),
                   forget.bias = mx.symbol.Variable(paste0(prefix, "l", i, "forget.bias")),
                   reset.weight = mx.symbol.Variable(paste0(prefix, "l", i, "reset.weight")),
                   reset.bias = mx.symbol.Variable(paste0(prefix, "l", i, "reset.bias")),
                   highway.weight = mx.symbol.Variable(paste0(prefix, "l", i, "highway.weight")),
                   highway.bias = mx.symbol.Variable(paste0(prefix, "l", i, "highway.bias")),
                   read.weight = mx.symbol.Variable(paste0(prefix, "l", i, "read.weight")),
                   read.bias = mx.symbol.Variable(paste0(prefix, "l", i, "read.bias")))
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
      data <- mx.symbol.SequenceReverse(data = data, use.sequence.length = T, axis = 0, sequence_length = seq.mask, name = paste0(prefix, "reverse"))  
    } else {
      data <- mx.symbol.SequenceReverse(data = data, use.sequence.length = F, axis = 0, name = paste0(prefix, "reverse"))
    }
  } else if (bidirectional) {
    if (masking) {
      data_rev <- mx.symbol.SequenceReverse(data = data, use.sequence.length = T, axis = 0, sequence_length = seq.mask, name = paste0(prefix, "reverse"))  
    } else {
      data_rev <- mx.symbol.SequenceReverse(data = data, use.sequence.length = F, axis = 0, name = paste0(prefix, "reverse"))
    }
  } 
  
  
  hidden <- list()
  last.hidden <- list()
  last.states <- list()
  
  for (i in 1:num_rnn_layer) {
    
    hidden <- list()
    param <- param.cells[[i]]
    
    in_proj <- mx.symbol.FullyConnected(data = data, weight = param$input.weight, bias = param$input.bias, flatten = F,
                                        num_hidden = num_hidden, name = paste0(prefix, "input", "_l", i))
    
    mem <- mx.symbol.FullyConnected(data = data, weight = param$mem.weight, bias = param$mem.bias, flatten = F,
                                    num_hidden = num_hidden, name = paste0(prefix, "mem", "_l", i))
    
    write <- mx.symbol.FullyConnected(data = data, weight = param$write.weight, bias = param$write.bias, flatten = F,
                                      num_hidden = num_hidden, name = paste0(prefix, "write", "_l", i))
    
    read <- mx.symbol.FullyConnected(data = data, weight = param$read.weight, bias = param$read.bias, flatten = F,
                                     num_hidden = num_hidden, name = paste0(prefix, "read", "_l", i))
    
    highway <- mx.symbol.FullyConnected(data = data, weight = param$highway.weight, bias = param$highway.bias, flatten = F,
                                        num_hidden = num_hidden, name = paste0(prefix, "highway", "_l", i))
    
    mem <- mx.symbol.relu(mem)
    write <- mx.symbol.tanh(write)
    read <- mx.symbol.sigmoid(read)
    highway <- mx.symbol.sigmoid(highway)
    
    in_proj <- mx.symbol.split(in_proj, num_outputs = seq_len, axis=0, squeeze.axis = F)
    mem <- mx.symbol.split(mem, num_outputs = seq_len, axis=0, squeeze.axis = F)
    write <- mx.symbol.split(write, num_outputs = seq_len, axis=0, squeeze.axis = F)
    read <- mx.symbol.split(read, num_outputs = seq_len, axis=0, squeeze.axis = F)
    highway <- mx.symbol.split(highway, num_outputs = seq_len, axis=0, squeeze.axis = F)
    
    for (seqidx in 1:seq_len) {
      
      if (seqidx == 1) {
        next.c <- write[[seqidx]] * mem[[seqidx]]
        next.h <-  highway[[seqidx]] * in_proj[[seqidx]]
      } else {
        next.c <- prev.c + write[[seqidx]] * mem[[seqidx]]
        next.h <- read[[seqidx]] * prev.c + highway[[seqidx]] * in_proj[[seqidx]]
      }
      prev.c <- next.c
      hidden <- c(hidden, next.h)
    }
    # concat hidden units - concat seq_len blocks of dimension num_hidden x batch.size -> 
    data <- mx.symbol.concat(hidden, num.args=seq_len, dim=0)
  }
  
  
  # RNN on reverse sequence
  if (bidirectional) {
    
    last.hidden.rev <- list()
    last.states.rev <- list()
    
    for (i in 1:num_rnn_layer) {
      
      hidden <- list()
      param <- param.cells[[i]]
      
      in_proj <- mx.symbol.FullyConnected(data = data_rev, weight = param$input.weight, bias = param$input.bias, flatten = F,
                                          num_hidden = num_hidden, name = paste0(prefix, "input", "_l", i))
      
      mem <- mx.symbol.FullyConnected(data = data_rev, weight = param$mem.weight, bias = param$mem.bias, flatten = F,
                                      num_hidden = num_hidden, name = paste0(prefix, "mem", "_l", i))
      
      write <- mx.symbol.FullyConnected(data = data_rev, weight = param$write.weight, bias = param$write.bias, flatten = F,
                                        num_hidden = num_hidden, name = paste0(prefix, "write", "_l", i))
      
      read <- mx.symbol.FullyConnected(data = data_rev, weight = param$read.weight, bias = param$read.bias, flatten = F,
                                       num_hidden = num_hidden, name = paste0(prefix, "read", "_l", i))
      
      highway <- mx.symbol.FullyConnected(data = data_rev, weight = param$highway.weight, bias = param$highway.bias, flatten = F,
                                          num_hidden = num_hidden, name = paste0(prefix, "highway", "_l", i))
      
      mem <- mx.symbol.relu(mem)
      write <- mx.symbol.tanh(write)
      read <- mx.symbol.sigmoid(read)
      highway <- mx.symbol.sigmoid(highway)
      
      in_proj <- mx.symbol.split(in_proj, num_outputs = seq_len, axis=0, squeeze.axis = F)
      mem <- mx.symbol.split(mem, num_outputs = seq_len, axis=0, squeeze.axis = F)
      write <- mx.symbol.split(write, num_outputs = seq_len, axis=0, squeeze.axis = F)
      read <- mx.symbol.split(read, num_outputs = seq_len, axis=0, squeeze.axis = F)
      highway <- mx.symbol.split(highway, num_outputs = seq_len, axis=0, squeeze.axis = F)
      
      for (seqidx in 1:seq_len) {
        
        if (seqidx == 1) {
          next.c <- write[[seqidx]] * mem[[seqidx]]
          next.h <-  highway[[seqidx]] * in_proj[[seqidx]]
        } else {
          next.c <- prev.c + write[[seqidx]] * mem[[seqidx]]
          next.h <- read[[seqidx]] * prev.c + highway[[seqidx]] * in_proj[[seqidx]]
        }
        prev.c <- next.c
        hidden <- c(hidden, next.h)
      }
      # concat hidden units - concat seq_len blocks of dimension num_hidden x batch.size -> 
      data_rev <- mx.symbol.concat(hidden, num.args=seq_len, dim=0)
    }
  }
  ### End of bidirectionnal RNN ###
  
  if (output_last_state) {
    out.states = mx.symbol.Group(unlist(last.states))
  }
  
  if (reverse_input) {
    if (masking) {
      data <- mx.symbol.SequenceReverse(data = data, use.sequence.length = T, axis = 0, sequence_length = seq.mask, name = paste0(prefix, "reverse_post"))  
    } else {
      data <- mx.symbol.SequenceReverse(data = data, use.sequence.length = F, axis = 0, name = paste0(prefix, "reverse_post"))
    }
  } else if (bidirectional) {
    if (masking) {
      data_rev <- mx.symbol.SequenceReverse(data = data_rev, use.sequence.length = T, axis = 0, sequence_length = seq.mask, name = paste0(prefix, "reverse_post"))  
    } else {
      data_rev <- mx.symbol.SequenceReverse(data = data_rev, use.sequence.length = F, axis = 0, name = paste0(prefix, "reverse_post"))
    }
    data <- data + data_rev
  }
  
  if (config=="seq-to-one") {
    
    if (masking) {
      mask <- mx.symbol.SequenceLast(data = data, use.sequence.length = T, axis = 0, sequence_length = seq.mask, name = paste0(prefix, "mask")) 
    } else mask <- mx.symbol.SequenceLast(data = data, use.sequence.length = F, axis = 0, name = paste0(prefix, "mask"))
    
    if (!is.null(loss_output)) {
      
      decode <- mx.symbol.FullyConnected(data = mask,
                                         weight = cls.weight,
                                         bias = cls.bias,
                                         num_hidden = num_decode,
                                         name = paste0(prefix, "decode"))
      
      out <- switch(loss_output,
                    softmax = mx.symbol.SoftmaxOutput(data=decode, label=label, use_ignore = !ignore_label == -1, ignore_label = ignore_label, name = paste0(prefix, "loss")),
                    linear = mx.symbol.LinearRegressionOutput(data=decode, label=label, name = paste0(prefix, "loss")),
                    logistic = mx.symbol.LogisticRegressionOutput(data=decode, label=label, paste0(prefix, name = "loss")),
                    MAE = mx.symbol.MAERegressionOutput(data=decode, label=label, paste0(prefix, name = "loss"))
      )
    } else out <- mask
    
  } else if (config=="one-to-one"){
    
    if (masking) {
      mask <- mx.symbol.SequenceMask(data = data, use.sequence.length = T, axis = 0, sequence_length = seq.mask, value = 0, name = paste0(prefix, "mask"))
    } else mask <- mx.symbol.identity(data = data, name = paste0(prefix, "mask"))
    
    mask = mx.symbol.swapaxes(data = mask, dim1 = 0, dim2 = 1, name = paste0(prefix, "swap_post"))
    
    if (!is.null(loss_output)) {
      
      mask <- mx.symbol.reshape(data = mask, shape = c(0, -1), reverse = TRUE)
      label <- mx.symbol.reshape(data = label, shape = c(-1))
      
      decode <- mx.symbol.FullyConnected(data = mask, weight = cls.weight, bias = cls.bias, num_hidden = num_decode, 
                                         flatten = T, name = paste0(prefix, "decode"))
      
      out <- switch(loss_output,
                    softmax = mx.symbol.SoftmaxOutput(data=decode, label=label, use_ignore = !ignore_label == -1, ignore_label = ignore_label, 
                                                      name = paste0(prefix, "loss")),
                    linear = mx.symbol.LinearRegressionOutput(data=decode, label=label, name = paste0(prefix, "loss")),
                    logistic = mx.symbol.LogisticRegressionOutput(data=decode, label=label, name = paste0(prefix, "loss")),
                    MAE = mx.symbol.MAERegressionOutput(data=decode, label=label, name = paste0(prefix, "loss"))
      )
    } else out <- mask
  }
  
  if (output_last_state) {
    return(mx.symbol.Group(c(out, out.states)))
  } else return(out)
}
