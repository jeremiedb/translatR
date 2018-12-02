
#' unroll representation of RNN running on non CUDA device with attention
#' 
#' @export
rnn.graph.unroll.decode <- function(mode = "train",
                                    encode,
                                    num_rnn_layer, 
                                    attn,
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
                                    label_name = "label") {
  
  
  # attention weights
  attn.weight <- mx.symbol.Variable(paste0(prefix, "attn.weight"))
  attn.bias <- mx.symbol.Variable(paste0(prefix, "attn.bias"))
  
  attn_init <- attn$init()
  attend <- attn$attend
  
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
    } else if (cell_type=="light") {
      cell <- list(input_weight = mx.symbol.Variable(paste0(prefix, "l", i, "_input_weight")),
                   proj_weight = mx.symbol.Variable(paste0(prefix, "l", i, "_proj_weight")),
                   proj_bias = mx.symbol.Variable(paste0(prefix, "l", i, "_proj_bias")),
                   gate_c_weight = mx.symbol.Variable(paste0(prefix, "l", i, "_gate_c_weight")),
                   gate_c_bias = mx.symbol.Variable(paste0(prefix, "l", i, "_gate_c_bias")),
                   gate_h_weight = mx.symbol.Variable(paste0(prefix, "l", i, "_gate_h_weight")),
                   gate_h_bias = mx.symbol.Variable(paste0(prefix, "l", i, "_gate_h_bias")),
                   write_weight = mx.symbol.Variable(paste0(prefix, "l", i, "_write_weight")),
                   write_bias = mx.symbol.Variable(paste0(prefix, "l", i, "_write_bias")),
                   proj_c_weight = mx.symbol.Variable(paste0(prefix, "l", i, "_proj_c_weight")),
                   proj_c_bias = mx.symbol.Variable(paste0(prefix, "l", i, "_proj_c_bias")),
                   gate_c_c_weight = mx.symbol.Variable(paste0(prefix, "l", i, "_gate_c_c_weight")),
                   gate_c_c_bias = mx.symbol.Variable(paste0(prefix, "l", i, "_gate_c_c_bias")),
                   gate_c_h_weight = mx.symbol.Variable(paste0(prefix, "l", i, "_gate_c_h_weight")),
                   gate_c_h_bias = mx.symbol.Variable(paste0(prefix, "l", i, "_gate_c_h_bias")),
                   write_c_weight = mx.symbol.Variable(paste0(prefix, "l", i, "_write_c_weight")),
                   write_c_bias = mx.symbol.Variable(paste0(prefix, "l", i, "_write_c_bias")))
    }
    return (cell)
  })
  
  if (cell_type=="lstm") {
    cell.symbol <- lstm.cell
  } else if (cell_type=="gru"){
    cell.symbol <- gru.cell
  } else if (cell_type=="straight"){
    cell.symbol <- straight.cell
  } else if (cell_type=="rich"){
    cell.symbol <- rich.cell
  } else if (cell_type=="light"){
    cell.symbol <- light.cell
  }
  
  
  # set label
  label <- mx.symbol.Variable(label_name)
  
  # embedding weights
  embed_ini_weight <- mx.symbol.Variable(paste0(prefix, "embed_ini_weight"))
  if (!is.null(num_embed)) embed_weight <- mx.symbol.Variable(paste0(prefix, "embed_weight"))
  
  # set inital input data
  data_feed <- mx.symbol.slice_axis(label, axis = 1, begin = 0, end = 1, name = paste0(prefix, "data_ini"))
  data_feed <- mx.symbol.reshape(data = data_feed, shape = -1)
  data_feed <- mx.symbol.Embedding(data = data_feed, input_dim = 1, weight = embed_ini_weight, 
                                    output_dim =  2 * num_hidden, name = paste0(prefix, "embed_ini"))
  
  if (masking) {
    seq.mask <- mxnet:::mx.varg.symbol.internal.not_equal_scalar(alist = list(data=label, scalar = 0))
    seq.mask <- mx.symbol.sum_axis(data = seq.mask, axis = 1)
  }
  
  # data for teacher
  data <- mx.symbol.split(label, num_outputs = seq_len, axis = 1, squeeze_axis = T)
  # Split labels with a right and lift offset to use as input and output of the decoder
  # data <- mx.symbol.slice_axis(label, axis = 1, begin=0, end=-1)
  # label = mx.symbol.slice_axis(label, axis=0, begin=1, end="None", name = "label_out")
  
  last.decode <- list()
  last.states <- list()
  
  for (seqidx in 1:(seq_len)) {
    
    hidden <- data_feed
    
    ### Attention
    # ctx_vector <- attn$attend_in(query = hidden, key = attn_init$key, value = attn_init$value, attn_init = attn_init)
    # # combine context vector with last hidden to form the attn vector
    # ctx_hidden <- mx.symbol.concat(data = c(hidden, ctx_vector), num.args = 2, dim = -1)
    # hidden <- mx.symbol.FullyConnected(data = ctx_hidden, num_hidden = num_hidden, weight=attn.weight, bias=attn.bias, no_bias=F) %>%
    #   mx.symbol.tanh()
    
    for (i in 1:num_rnn_layer) {
      
      if (seqidx==1) prev.state <- init.state[[i]] else 
        prev.state <- last.states[[i]]
      
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
    ctx_vector <- attn$attend(query = hidden, key = attn_init$key, value = attn_init$value, attn_init = attn_init)
    # combine context vector with last hidden to form the attn vector
    ctx_hidden <- mx.symbol.concat(data = c(hidden, ctx_vector), num.args = 2, dim = -1)
    attention <- mx.symbol.FullyConnected(data = ctx_hidden, num_hidden = num_hidden, weight=attn.weight, bias=attn.bias, no_bias=F) %>%
      mx.symbol.tanh()
    
    # data_feed <- mx.symbol.concat(data = c(hidden, attention), num.args = 2, dim = -1)
    
    decode <- mx.symbol.FullyConnected(data = attention,
                                       weight = cls.weight,
                                       bias = cls.bias,
                                       num_hidden = num_decode,
                                       flatten = T)
    
    # Aggregate outputs from each timestep
    last.decode <- c(last.decode, decode)
    
    if (seqidx < seq_len) {
      
      if (mode == "argmax") {
        # Argmax token selection
        sample <- mx.symbol.argmax(decode, axis = 1, keepdims = F)
        sample <- mx.symbol.Embedding(data = sample, input_dim = input_size, weight = embed_weight, output_dim = num_embed)
      } else if (mode == "sample") {
        # Multinomial token sample
        sample <- mx.symbol.softmax(data = decode, axis = -1)
        sample <- mx.symbol.sample_multinomial(data = sample, get_prob = F)
        sample <- mx.symbol.Embedding(data = sample, input_dim = input_size, weight = embed_weight, output_dim = num_embed)
      } else if (mode == "teacher") {
        sample <- mx.symbol.Embedding(data = mx.symbol.BlockGrad(data[[seqidx]]), input_dim = input_size, weight = embed_weight, output_dim = num_embed)
      } else if (mode == "ctx") {
        sample <- ctx_hidden
      } else if (mode == "attention") {
        sample <- attention
      }
      data_feed <- mx.symbol.concat(data = c(hidden, sample), num.args = 2, dim = -1)
    }
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
