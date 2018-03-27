ConvComp.graph.encode <- function(input_size = NULL,
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
  
  data <- mx.symbol.Variable(data_name)
  label <- mx.symbol.Variable(label_name)
  
  # embeding layer
  if (!is.null(num_embed)) embed.weight <- mx.symbol.Variable(paste0(prefix, "embed.weight"))
  
  if (masking) {
    seq.mask <- mxnet:::mx.varg.symbol.internal.not_equal_scalar(alist = list(data=data, scalar = 0))
    seq.mask <- mx.symbol.sum_axis(data = seq.mask, axis = 1)
  }
  
  if (!is.null(num_embed)) {
    data <- mx.symbol.Embedding(data = data, input_dim = input_size,
                                weight=embed.weight, output_dim = num_embed, name = paste0(prefix, "embed"))
  }
  
  data <- mx.symbol.swapaxes(data = data, dim1 = 1, dim2 = 2)
  data <- mx.symbol.expand_dims(data=data, axis=-1)
  
  # U-net encoding
  level1_conv <- mx.symbol.Convolution(data=data, kernel=c(1, 5), stride=c(1,1), pad=c(0,2), num_filter=num_hidden) %>% 
    mx.symbol.BatchNorm() %>%
    mx.symbol.relu
  # level1_gate <- mx.symbol.Convolution(data=data, kernel=c(1, 3), stride=c(1,1), pad=c(0,1), num_filter=num_hidden) %>% mx.symbol.sigmoid
  # level1 <- level1_conv * level1_gate + (1-level1_gate) * data
  
  # level1$get.internals()$infer.shape(list(data = c(24, 16)))$out.shapes
  
  level2 <- mx.symbol.Pooling(data=level1_conv, global.pool=F, pool.type="max" , kernel=c(1,2), stride=c(1,2), pad=c(0,0))
  level2_conv <- mx.symbol.Convolution(data=level2, kernel=c(1, 3), stride=c(1,1), pad=c(0,1), num_filter=num_hidden+50) %>% 
    mx.symbol.BatchNorm() %>%
    mx.symbol.relu
  # level2_gate <- mx.symbol.Convolution(data=level2, kernel=c(1, 3), stride=c(1,1), pad=c(0,1), num_filter=num_hidden) %>% mx.symbol.sigmoid
  # level2 <- level2_conv * level2_gate + (1 - level2_gate) * level2
  
  level3 <- mx.symbol.Pooling(data=level2_conv, global.pool=F, pool.type="max" , kernel=c(1,1), stride=c(1,2), pad=c(0,0))
  level3_conv <- mx.symbol.Convolution(data=level3, kernel=c(1, 3), stride=c(1,1), pad=c(0,1), num_filter=num_hidden+100) %>% 
    mx.symbol.BatchNorm() %>%
    mx.symbol.relu
  # level3_gate <- mx.symbol.Convolution(data=level3, kernel=c(1, 3), stride=c(1,1), pad=c(0,1), num_filter=num_hidden) %>% mx.symbol.sigmoid
  # level3 <- level3_conv * level3_gate + (1 - level3_gate) * level3

  # reshape into feature X batch X seq
  concat <- mx.symbol.reshape(data = level3_conv, shape = c(0,0,0))
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
                    softmax = mx.symbol.SoftmaxOutput(data=decode, label=label, use_ignore = !ignore_label == -1, ignore_label = ignore_label, 
                                                      name = paste0(prefix, "loss")),
                    linear = mx.symbol.LinearRegressionOutput(data=decode, label=label, name = paste0(prefix, "loss")),
                    logistic = mx.symbol.LogisticRegressionOutput(data=decode, label=label, name = paste0(prefix, "loss")),
                    MAE = mx.symbol.MAERegressionOutput(data=decode, label=label, name = paste0(prefix, "loss"))
      )
    } else out <- mask
  }
  
  return(out)
}
