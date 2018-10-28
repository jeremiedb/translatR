
#' unroll representation of RNN running on non CUDA device with attention
#' 
#' @export
rnn.graph.unroll.decode <- function(num_rnn_layer, 
                                    attn_init,
                                    attend,
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
                                    label_name = "label") {
  
  
  # attention weights
  attn.weight <- mx.symbol.Variable(paste0(prefix, "attn.weight"))
  attn.bias <- mx.symbol.Variable(paste0(prefix, "attn.bias"))
  
  # Initial attn parameters - shape: num_hidden_encode
  # decode.init.weight = mx.symbol.Variable("decode.init.weight")
  # attn.init.weight <- mx.symbol.Variable(paste0(prefix, "attn.init.weight"))
  # attn.init.bias <- mx.symbol.Variable(paste0(prefix, "attn.init.bias"))
  
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
  
  
  # set label
  label <- mx.symbol.Variable(label_name)
  
  # embedding weights
  embed_ini_weight <- mx.symbol.Variable("embed_ini_weight")
  if (!is.null(num_embed)) embed_weight <- mx.symbol.Variable("embed_weight")
  
  # set inital input data
  data <- mx.symbol.slice_axis(label, axis = 1, begin = 0, end = 1, name = "data_ini")
  data <- mx.symbol.reshape(data = data, shape = -1)
  hidden <- mx.symbol.Embedding(data = data, input_dim = 1, weight = embed_ini_weight, 
                                output_dim =  num_hidden + num_embed, name = paste0(prefix, "embed"))
  
  if (masking) {
    seq.mask <- mxnet:::mx.varg.symbol.internal.not_equal_scalar(alist = list(data=label, scalar = 0))
    seq.mask <- mx.symbol.sum_axis(data = seq.mask, axis = 1)
  }
  
  # Split labels with a right and lift offset to use as input and output of the decoder
  # data = mx.symbol.slice_axis(label, axis=0, begin=0, end=-1, name = "label_in")
  # label = mx.symbol.slice_axis(label, axis=0, begin=1, end="None", name = "label_out")
  
  last.decode <- list()
  last.states <- list()
  
  for (seqidx in 1:(seq_len)) {
    
    for (i in 1:num_rnn_layer) {
      
      if (seqidx==1) prev.state <- init.state[[i]] else prev.state <- last.states[[i]]
      
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
    
    ### Attention
    ctx_vector <- attn$attend(query = hidden, key = attn_init$key, value = attn_init$value, query_weight = attn_init$query_weight)
    
    # combine context vector with last hidden to form the attn vector
    hidden <- mx.symbol.concat(data = c(hidden, ctx_vector), num.args = 2, dim = -1)
    hidden <- mx.symbol.FullyConnected(data = hidden, num_hidden = num_hidden, weight=attn.weight, bias=attn.bias, no_bias=F) %>% 
      mx.symbol.tanh()
    
    decode <- mx.symbol.FullyConnected(data = hidden,
                                       weight = cls.weight,
                                       bias = cls.bias,
                                       num_hidden = num_decode,
                                       name = paste0(prefix, "decode"),
                                       flatten = T)
    
    # Sample selected token - argmax or multinomial sample
    sample <- mx.symbol.argmax(decode, axis = 1, keepdims = F)
    sample <- mx.symbol.Embedding(data = sample, input_dim = input_size, weight = embed_weight, 
                                  output_dim = num_embed, name = paste0(prefix, "embed"))
    hidden <- mx.symbol.concat(data = c(hidden, sample), num.args = 2, dim = -1)
    
    # Aggregate outputs from each timestep
    last.decode <- c(last.decode, decode)
    
  }
  
  # concat hidden units - concat seq_len blocks of dimension num_decode x batch.size
  concat <- mx.symbol.concat(data = last.decode, num.args = seq_len, dim = 0, name = paste0(prefix, "concat"))
  concat <- mx.symbol.reshape(data = concat, shape = c(num_decode, -1, seq_len), name = paste0(prefix, "rnn_reshape"))
  
  if (masking) mask <- mx.symbol.SequenceMask(data = concat, use.sequence.length = T, sequence_length = seq.mask, value = 0, name = paste0(prefix, "mask")) else
    mask <- mx.symbol.identity(data = concat, name = paste0(prefix, "mask"))
  
  mask <- mx.symbol.swapaxes(data = mask, dim1 = 0, dim2 = 1, name = paste0(prefix, "swap_post"))
  
  mask <- mx.symbol.reshape(data = mask, shape = c(num_decode, -1))
  label <- mx.symbol.reshape(data = label, shape = c(-1))
  
  out <- switch(loss_output,
                softmax = mx.symbol.SoftmaxOutput(data=mask, label=label, use_ignore = !ignore_label == -1, ignore_label = ignore_label, 
                                                  normalization = "valid", 
                                                  name = paste0(prefix, "loss")),
                linear = mx.symbol.LinearRegressionOutput(data=decode, label=label, name = paste0(prefix, "loss")),
                logistic = mx.symbol.LogisticRegressionOutput(data=decode, label=label, name = paste0(prefix, "loss")),
                MAE = mx.symbol.MAERegressionOutput(data=decode, label=label, name = paste0(prefix, "loss")))
  return(out)
}
