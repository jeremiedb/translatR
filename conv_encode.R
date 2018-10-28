conv.graph.encode <- function(input_size = NULL,
                              num_embed = NULL, 
                              num_hidden,
                              num_decode,
                              dropout = 0,
                              ignore_label = -1,
                              loss_output = NULL, 
                              masking = F, 
                              config,
                              prefix = "",
                              data_name = "data",
                              label_name = "label") {
  
  cls.weight <- mx.symbol.Variable(paste0(prefix, "cls.weight"))
  cls.bias <- mx.symbol.Variable(paste0(prefix, "cls.bias"))
  
  # [seq, batch]
  data <- mx.symbol.Variable(data_name)
  
  # [seq, batch]
  label <- mx.symbol.Variable(label_name)
  
  # embeding layer
  if (!is.null(num_embed)) embed.weight <- mx.symbol.Variable(paste0(prefix, "embed.weight"))
  
  if (masking) {
    # [seq, batch]
    seq_mask <- mxnet:::mx.varg.symbol.internal.not_equal_scalar(alist = list(data = data, scalar = 0))
    # [seq, 1, batch]
    seq_mask <- mx.symbol.expand_dims(seq_mask, axis = 1)
    # seq_mask <- mx.symbol.sum_axis(data = seq_mask, axis = 1)
  }
  
  # [embed, seq, batch]
  if (!is.null(num_embed)) {
    data <- mx.symbol.Embedding(data = data, input_dim = input_size,
                                weight=embed.weight, output_dim = num_embed, name = paste0(prefix, "embed"))
  }
  
  # [seq, embed, batch] - For convolutions to apply on the embed channels
  data <- mx.symbol.swapaxes(data = data, dim1 = 1, dim2 = 2)
  
  # Conv transform
  conv_factory <- function(data, kernel, stride, pad, num_filter, residual = F, num_filter_res = NULL) {
    conv <- mx.symbol.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter)
    gate <- mx.symbol.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter)
    out <- conv * mx.symbol.sigmoid(gate)
    
    if (residual) {
      out = out + data
      if (!is.null(num_filter_res)) {
        out <- mx.symbol.Convolution(data=out, kernel=c(1), stride=c(1), pad=c(0), num_filter=num_filter_out)
      }
    }
    return(out)
  }
  
  # Conv residual transform
  conv_res_factory <- function(data, 
                               depth = 1, 
                               kernel = 3, 
                               stride = 1, 
                               pad = 1, 
                               num_filter,
                               out_proj = F, 
                               num_filter_proj = NULL) {
    
    # Gate
    gate <- mx.symbol.Convolution(data = data, kernel = kernel, stride = stride, pad=pad, num_filter = num_filter) %>% 
      mx.symbol.sigmoid
    
    conv <- data
    for (i in seq_along(depth)) {
      conv <- mx.symbol.Convolution(data=conv, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter) %>% 
        mx.symbol.BatchNorm() %>%
        mx.symbol.relu()
    }
    
    data <- gate * data + (1 - gate) * conv
    
    if (out_proj) {
      if (!is.null(num_filter_proj)) {
        data <- mx.symbol.Convolution(data = data, kernel = 1, stride = 1, pad = 0, num_filter = num_filter_proj) %>% 
          mx.symbol.relu()
      }
    }
    return(data)
  }
  
  # [seq, features, batch]
  data <- conv_res_factory(data = data, depth = 1, kernel = 3, stride = 1, pad = 1, num_filter = num_hidden, out_proj = F, num_filter_proj = num_hidden)
  data <- conv_res_factory(data = data, depth = 1, kernel = 3, stride = 1, pad = 1, num_filter = num_hidden, out_proj = F, num_filter_proj = num_hidden)
  data <- conv_res_factory(data = data, depth = 1, kernel = 3, stride = 1, pad = 1, num_filter = num_hidden, out_proj = F, num_filter_proj = num_hidden)
  
  # [features, seq, batch]
  # data <- mx.symbol.swapaxes(data = data, dim1 = 1, dim2 = 2)
  
  # [seq, features, batch] X [seq, 1, batch] -> [seq, features, batch]
  if (masking) {
    data <- mx.symbol.broadcast_mul(data, seq_mask)
    # mask <- mx.symbol.SequenceMask(data = concat, use.sequence.length = T, sequence_length = seq_mask, value = 0, name = paste0(prefix, "mask"))
  }
  
  # [seq, features, batch]
  return(data)
}
