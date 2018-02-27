conv.graph.decode <- function(encode, 
                              input_size = NULL,
                              num_embed = NULL, 
                              num_hidden,
                              num_decode,
                              dropout = 0,
                              ignore_label = -1,
                              loss_output = NULL, 
                              masking = F, 
                              config,
                              prefix = "",
                              label_name = "label") {
  
  cls.weight <- mx.symbol.Variable(paste0(prefix, "cls.weight"))
  cls.bias <- mx.symbol.Variable(paste0(prefix, "cls.bias"))
  
  label <- mx.symbol.Variable(label_name)
  
  if (masking) {
    seq.mask <- mxnet:::mx.varg.symbol.internal.not_equal_scalar(alist = list(data=label, scalar = 0))
    seq.mask <- mx.symbol.sum_axis(data = seq.mask, axis = 1)
  }
  
  data <- mx.symbol.swapaxes(data = encode, dim1 = 1, dim2 = 2)
  data <- mx.symbol.expand_dims(data=data, axis=-1)
  
  # Conv transform
  conv_factory <- function(data, kernel, stride, pad, num_filter, residual = F, num_filter_res = NULL) {
    conv <- mx.symbol.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter)
    gate <- mx.symbol.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter)
    out <- conv * mx.symbol.sigmoid(gate)
    
    if (residual) {
      out = out + data
      if (!is.null(num_filter_res)) {
        out <- mx.symbol.Convolution(data=out, kernel=c(1,1), stride=c(1,1), pad=c(0,0), num_filter=num_filter_out)
      }
    }
    return(out)
  }
  
  # Conv residual transform
  conv_res_factory <- function(data, kernel, stride, pad, num_filter, out_proj = F, num_filter_proj = NULL) {
    conv <- mx.symbol.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter) %>% 
      # mx.symbol.BatchNorm() %>% 
      mx.symbol.relu()
    gate <- mx.symbol.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter) %>% 
      mx.symbol.sigmoid
    out <- gate * data + (1-gate) * conv
    
    if (out_proj) {
      if (!is.null(num_filter_proj)) {
        out <- mx.symbol.Convolution(data=out, kernel=c(1,1), stride=c(1,1), pad=c(0,0), num_filter=num_filter_proj) %>% 
          # mx.symbol.BatchNorm() %>% 
          mx.symbol.relu()
      }
    }
    return(out)
  }
  
  # data <- conv_factory(data = data, kernel = c(1,3), stride = c(1,1), pad = c(0,1), num_filter = num_hidden, residual = F, num_filter_res = NULL)
  # data <- conv_factory(data = data, kernel = c(1,3), stride = c(1,1), pad = c(0,1), num_filter = num_hidden, residual = F, num_filter_res = NULL)
  
  data <- conv_res_factory(data = data, kernel = c(1,3), stride = c(1,1), pad = c(0,1), num_filter = num_hidden, out_proj = F, num_filter_proj = num_hidden)
  data <- conv_res_factory(data = data, kernel = c(1,3), stride = c(1,1), pad = c(0,1), num_filter = num_hidden, out_proj = T, num_filter_proj = num_hidden)
  
  # data <- mx.symbol.Convolution(data = data, kernel = c(1,3), stride = c(1,1), pad = c(0,1), num_filter=num_hidden) %>% 
  #   mx.symbol.BatchNorm() %>% 
  #   mx.symbol.relu()
  
  concat <- mx.symbol.reshape(data = data, shape = c(0,0,0))
  concat <- mx.symbol.swapaxes(data = concat, dim1 = 1, dim2 = 2)
  concat <- mx.symbol.swapaxes(data = concat, dim1 = 0, dim2 = 1)
  
  if (config=="seq-to-one") {
    
    if (masking) {
      mask <- mx.symbol.SequenceLast(data=concat, use.sequence.length = T, sequence_length = seq.mask, name = paste0(prefix, "mask")) 
    } else mask <- mx.symbol.SequenceLast(data=concat, use.sequence.length = F, name = paste0(prefix, "mask"))
    
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
      mask <- mx.symbol.SequenceMask(data = concat, use.sequence.length = T, sequence_length = seq.mask, value = 0, name = paste0(prefix, "mask"))
    } else mask <- mx.symbol.identity(data = concat, name = paste0(prefix, "mask"))
    
    mask = mx.symbol.swapaxes(data = mask, dim1 = 0, dim2 = 1, name = paste0(prefix, "swap_post"))
    
    if (!is.null(loss_output)) {
      
      mask <- mx.symbol.reshape(data = mask, shape = c(0, -1), reverse = TRUE)
      label <- mx.symbol.reshape(data = label, shape = c(-1))
      
      decode <- mx.symbol.FullyConnected(data = mask, weight = cls.weight, bias = cls.bias, num_hidden = num_decode, 
                                         flatten = T, name = paste0(prefix, "decode"))
      
      out <- switch(loss_output,
                    softmax = mx.symbol.SoftmaxOutput(data=decode, label=label, 
                                                      use_ignore = !ignore_label == -1, ignore_label = ignore_label,
                                                      normalization = "valid",
                                                      name = paste0(prefix, "loss")),
                    linear = mx.symbol.LinearRegressionOutput(data=decode, label=label, name = paste0(prefix, "loss")),
                    logistic = mx.symbol.LogisticRegressionOutput(data=decode, label=label, name = paste0(prefix, "loss")),
                    MAE = mx.symbol.MAERegressionOutput(data=decode, label=label, name = paste0(prefix, "loss"))
      )
    } else out <- mask
  }
  
  return(out)
}