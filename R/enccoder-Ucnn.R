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
    # [seq, batch]
    seq_mask <- mxnet:::mx.varg.symbol.internal.not_equal_scalar(alist = list(data=data, scalar = 0))
    # [seq, 1, batch]
    seq_mask <- mx.symbol.expand_dims(seq_mask, axis = 1)
    # seq_mask <- mx.symbol.sum_axis(data = seq_mask, axis = 1)
  }
  
  if (!is.null(num_embed)) {
    data <- mx.symbol.Embedding(data = data, input_dim = input_size,
                                weight=embed.weight, output_dim = num_embed, name = paste0(prefix, "embed"))
  }
  
  data <- mx.symbol.swapaxes(data = data, dim1 = 1, dim2 = 2)
  
  # U-net encoding
  level1_conv <- mx.symbol.Convolution(data=data, kernel = 3, stride=1, pad=1, num_filter=num_hidden) %>% 
    mx.symbol.BatchNorm() %>%
    mx.symbol.relu
  # level1_gate <- mx.symbol.Convolution(data=data, kernel=c(1, 3), stride=c(1,1), pad=c(0,1), num_filter=num_hidden) %>% mx.symbol.sigmoid
  # level1 <- level1_conv * level1_gate + (1-level1_gate) * data
  
  # level1$get.internals()$infer.shape(list(data = c(24, 16)))$out.shapes
  
  level2 <- mx.symbol.Pooling(data=level1_conv, global.pool=F, pool.type="max" , kernel=2, stride=2, pad=0)
  level2_conv <- mx.symbol.Convolution(data=level2, kernel=3, stride=1, pad=1, num_filter=num_hidden + 64) %>% 
    mx.symbol.BatchNorm() %>%
    mx.symbol.relu
  # level2_gate <- mx.symbol.Convolution(data=level2, kernel=c(1, 3), stride=c(1,1), pad=c(0,1), num_filter=num_hidden) %>% mx.symbol.sigmoid
  # level2 <- level2_conv * level2_gate + (1 - level2_gate) * level2
  
  level3 <- mx.symbol.Pooling(data=level2_conv, global.pool=F, pool.type="max" , kernel=2, stride=2, pad=0)
  level3_conv <- mx.symbol.Convolution(data=level3, kernel=3, stride=1, pad=1, num_filter=num_hidden + 128) %>% 
    mx.symbol.BatchNorm() %>%
    mx.symbol.relu
  # level3_gate <- mx.symbol.Convolution(data=level3, kernel=c(1, 3), stride=c(1,1), pad=c(0,1), num_filter=num_hidden) %>% mx.symbol.sigmoid
  # level3 <- level3_conv * level3_gate + (1 - level3_gate) * level3
  
  
  # if (masking) {
  #   data <- mx.symbol.broadcast_mul(data, seq_mask)
  #   # mask <- mx.symbol.SequenceMask(data = concat, use.sequence.length = T, sequence_length = seq_mask, value = 0, name = paste0(prefix, "mask"))
  # }
  
  return(level3_conv)
}
