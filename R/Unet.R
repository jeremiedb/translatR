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
  }
  
  # [embed, seq, batch]
  if (!is.null(num_embed)) {
    data <- mx.symbol.Embedding(data = data, input_dim = input_size,
                                weight=embed.weight, output_dim = num_embed, name = paste0(prefix, "embed"))
  }
  
  # [seq, embed, batch] - For convolutions to apply on the embed channels
  data <- mx.symbol.swapaxes(data = data, dim1 = 1, dim2 = 2)
  
  # variational dropout
  data <- mx.symbol.Dropout(data = data, p = dropout)
  
  # Conv residual transform
  conv_res <- function(data, 
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
    for (i in seq_len(depth)) {
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
  
  
  # Conv residual transform
  dilated_conv <- function(data, num_filter) {
    
    # Gate
    layer_1 <- mx.symbol.Convolution(data = data, kernel=3, stride=1, pad=1, dilate=1, num_filter=num_filter) %>% 
      mx.symbol.BatchNorm() %>%
      mx.symbol.relu()
    
    layer_2 <- mx.symbol.Convolution(data = layer_1, kernel=3, stride=1, pad=3, dilate=3, num_filter=num_filter) %>% 
      mx.symbol.BatchNorm() %>%
      mx.symbol.relu()
    
    concat <- mx.symbol.concat(c(data, layer_1, layer_2), num.args = 2, dim = 1)
    proj <- mx.symbol.Convolution(data = concat, kernel=1, stride=1, pad=0, dilate=1, num_filter=num_filter)
    
    return(proj)
  }
  
  # [seq, features, batch]
  data <- conv_res(data = data, depth = 1, kernel = 5, stride = 1, pad = 2, num_filter = num_hidden, out_proj = F, num_filter_proj = num_hidden)
  data <- dilated_conv(data = data, num_filter = num_hidden)
  data <- conv_res(data = data, depth = 1, kernel = 3, stride = 1, pad = 1, num_filter = num_hidden, out_proj = F, num_filter_proj = num_hidden)
  data <- dilated_conv(data = data, num_filter = num_hidden)
  data <- conv_res(data = data, depth = 2, kernel = 3, stride = 1, pad = 1, num_filter = num_hidden, out_proj = F, num_filter_proj = num_hidden)
  data <- dilated_conv(data = data, num_filter = num_hidden)
  data <- conv_res(data = data, depth = 1, kernel = 3, stride = 1, pad = 1, num_filter = num_hidden, out_proj = F, num_filter_proj = num_hidden)
  
  data <- mx.symbol.Convolution(data = data, kernel = 3, stride = 1, pad = 1, num_filter=num_hidden) %>% 
    mx.symbol.BatchNorm() %>%
    mx.symbol.relu()
  
  # [seq, features, batch] -> [features, seq, batch]
  data <- mx.symbol.swapaxes(data = level3_up, dim1 = 1, dim2 = 2)
  
  # [features, seq, batch] -> [features, seq * batch]
  data <- mx.symbol.reshape(data = data, shape = c(num_decode, -1))
  label <- mx.symbol.reshape(data = label, shape = c(-1))
  
  decode <- mx.symbol.FullyConnected(data = data, weight = cls.weight, bias = cls.bias, num_hidden = num_decode, 
                                     flatten = T, name = paste0(prefix, "decode"))
  
  decode <- mx.symbol.reshape(data = decode, shape = c(num_decode, -1))
  label <- mx.symbol.reshape(data = label, shape = c(-1))
  
  out <- mx.symbol.SoftmaxOutput(data=decode, label=label, use_ignore=!ignore_label == -1, ignore_label = ignore_label, 
                                 normalization = "valid", 
                                 name = paste0(prefix, "loss"))
  return(out)
}
