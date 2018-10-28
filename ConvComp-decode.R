ConvComp.graph.decode <- function(encode, 
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

  # if (masking) {
  #   # [seq, batch]
  #   seq_mask <- mxnet:::mx.varg.symbol.internal.not_equal_scalar(alist = list(data = label, scalar = 0))
  #   # [seq, 1, batch]
  #   seq_mask <- mx.symbol.expand_dims(seq_mask, axis = 1)
  #   # seq_mask <- mx.symbol.sum_axis(data = seq_mask, axis = 1)
  # }
  
  # encode <- mx.symbol.swapaxes(data = encode, dim1 = 1, dim2 = 2)
  # encode <- mx.symbol.expand_dims(data=encode, axis=-1)
  
  level1_up = mx.symbol.Deconvolution(encode, kernel = 2, pad = 0, stride = 2, num_filter = num_hidden + 64) %>% mx.symbol.relu()
  # level1_up = mx.symbol.UpSampling(encode, scale = 2, sample_type = "nearest", num_args = 1)
  level1_up_conv <- mx.symbol.Convolution(data=level1_up, kernel=3, stride=1, pad=1, num_filter=num_hidden + 64) %>% 
    mx.symbol.BatchNorm() %>%
    mx.symbol.relu()
  # level1_up_gate <- mx.symbol.Convolution(data=level1_up, kernel=c(1, 3), stride=c(1,1), pad=c(0,1), num_filter=num_hidden) %>% mx.symbol.sigmoid
  # level1_up <- level1_up_conv * level1_up_gate + (1-level1_up_gate) * level1_up
  
  # level2_up$get.internals()$infer.shape(list(data = c(24, 16)))$out.shapes
  
  level2_up = mx.symbol.Deconvolution(level1_up_conv, kernel = 2, pad = 0, stride = 2, num_filter = num_hidden + 32) %>% mx.symbol.relu()
  # level2_up = mx.symbol.UpSampling(level1_up_conv, scale = 2, sample_type = "nearest", num_args = 1)
  level2_up_conv <- mx.symbol.Convolution(data=level2_up, kernel=3, stride=c(1,1), pad=1, num_filter=num_hidden + 32) %>% 
    mx.symbol.BatchNorm() %>%
    mx.symbol.relu()
  # level2_up_gate <- mx.symbol.Convolution(data=level2_up, kernel=c(1, 3), stride=c(1,1), pad=c(0,1), num_filter=num_hidden) %>% mx.symbol.sigmoid
  # level2_up <- level2_up_conv * level2_up_gate + (1-level2_up_gate) * level2_up
  
  level3_up <- mx.symbol.Convolution(data=level2_up_conv, kernel=3, stride=1, pad=1, num_filter=num_hidden) %>% 
    mx.symbol.BatchNorm() %>%
    mx.symbol.relu()
  
  
  if (!is.null(loss_output)) {
    
    # [seq, features, batch] -> [features, seq, batch]
    data <- mx.symbol.swapaxes(data = level3_up, dim1 = 1, dim2 = 2)
    
    # [features, seq, batch] -> [features, seq * batch]
    data <- mx.symbol.reshape(data = data, shape = c(0, -1), reverse = TRUE)
    label <- mx.symbol.reshape(data = label, shape = c(-1))
    
    decode <- mx.symbol.FullyConnected(data = data, weight = cls.weight, bias = cls.bias, num_hidden = num_decode, 
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
  } else out <- data
  return(out)
}
