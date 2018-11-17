# LSTM cell symbol
lstm.cell <- function(num_hidden, indata, prev.state, param, seqidx, layeridx, dropout = 0, prefix = "") {
  
  if (dropout > 0 && layeridx > 1) 
    indata <- mx.symbol.Dropout(data = indata, p = dropout)
  
  i2h <- mx.symbol.FullyConnected(data = indata, weight = param$i2h.weight, bias = param$i2h.bias, 
                                  num_hidden = num_hidden * 4, name = paste0(prefix, "t", seqidx, ".l", layeridx, ".i2h"))
  
  if (!is.null(prev.state)) {
    h2h <- mx.symbol.FullyConnected(data = prev.state$h, weight = param$h2h.weight, 
                                    bias = param$h2h.bias, num_hidden = num_hidden * 4, 
                                    name = paste0(prefix, "t", seqidx, ".l", layeridx, ".h2h"))
    gates <- i2h + h2h
  } else {
    gates <- i2h
  }
  
  split.gates <- mx.symbol.split(gates, num.outputs = 4, axis = 1, squeeze.axis = F, 
                                 name = paste0(prefix, "t", seqidx, ".l", layeridx, ".slice"))
  
  in.gate <- mx.symbol.Activation(split.gates[[1]], act.type = "sigmoid")
  in.transform <- mx.symbol.Activation(split.gates[[2]], act.type = "tanh")
  forget.gate <- mx.symbol.Activation(split.gates[[3]], act.type = "sigmoid")
  out.gate <- mx.symbol.Activation(split.gates[[4]], act.type = "sigmoid")
  
  if (is.null(prev.state)) {
    next.c <- in.gate * in.transform
  } else {
    next.c <- (forget.gate * prev.state$c) + (in.gate * in.transform)
  }
  
  next.h <- out.gate * mx.symbol.Activation(next.c, act.type = "tanh")
  
  return(list(h = next.h, c = next.c))
}



# light cell symbol
light.cell <- function(num_hidden, indata, prev.state, param, seqidx, layeridx, dropout, prefix = "") {
  
  if (layeridx == 1) {
    indata <- mx.symbol.FullyConnected(data = indata, weight = param$input_weight, no_bias = T,
                                       num_hidden = num_hidden, name = paste0(prefix, "input", "_l", layeridx, "_s", seqidx))
    if (dropout > 0) indata <- mx.symbol.Dropout(data = indata, p = dropout)
  }
  
  gate_c <- mx.symbol.FullyConnected(data = indata, weight = param$gate_c_weight, bias = param$gate_c_bias,
                                     num_hidden = num_hidden, name = paste0(prefix, "gate_c", "_l", layeridx, "_s", seqidx))
  
  gate_h <- mx.symbol.FullyConnected(data = indata, weight = param$gate_h_weight, bias = param$gate_h_bias,
                                     num_hidden = num_hidden, name = paste0(prefix, "gate_h", "_l", layeridx, "_s", seqidx))
  
  proj <- mx.symbol.FullyConnected(data = indata, weight = param$proj_weight, bias = param$proj_bias,
                                   num_hidden = num_hidden, name = paste0(prefix, "proj", "_l", layeridx, "_s", seqidx))
  
  write <- mx.symbol.FullyConnected(data = indata, weight = param$write_weight, bias = param$write_bias,
                                    num_hidden = num_hidden, name = paste0(prefix, "write", "_l", layeridx, "_s", seqidx))
  
  if (!is.null(prev.state)) {
    gate_c_c <- mx.symbol.FullyConnected(data = prev.state$c, weight = param$gate_c_c_weight, bias = param$gate_c_c_bias,
                                         num_hidden = num_hidden, name = paste0(prefix, "gate_c_c", "_l", layeridx, "_s", seqidx))
    
    gate_c_h <- mx.symbol.FullyConnected(data = prev.state$c, weight = param$gate_c_h_weight, bias = param$gate_c_h_bias,
                                         num_hidden = num_hidden, name = paste0(prefix, "gate_c_h", "_l", layeridx, "_s", seqidx))
    
    proj_c <- mx.symbol.FullyConnected(data = prev.state$c, weight = param$proj_c_weight, bias = param$proj_c_bias,
                                       num_hidden = num_hidden, name = paste0(prefix, "proj_c", "_l", layeridx, "_s", seqidx))
    
    write_c <- mx.symbol.FullyConnected(data = prev.state$c, weight = param$write_c_weight, bias = param$write_c_bias,
                                        num_hidden = num_hidden, name = paste0(prefix, "write_c", "_l", layeridx, "_s", seqidx))
    
    gate_c <- gate_c + gate_c_c
    gate_h <- gate_h + gate_c_h
    proj <- proj + proj_c
    write <- write + write_c
  }
  
  gate_c <- mx.symbol.sigmoid(gate_c)
  gate_h <- mx.symbol.sigmoid(gate_h)
  write <- mx.symbol.tanh(write)
  
  if (is.null(prev.state)) {
    next_c <- gate_c * write
  } else {
    next_c <- gate_c * write + (1 - gate_c) * prev.state$c
  }
  
  next_h <- mx.symbol.tanh(proj)
  
  return(list(h = next_h, c = next_c))
}


# Straight cell symbol
straight.cell <- function(num_hidden, indata, prev.state, param, seqidx, layeridx, dropout, prefix = "") {
  
  if (dropout > 0 && layeridx > 1) 
    indata <- mx.symbol.Dropout(data = indata, p = dropout)
  
  proj_full <- mx.symbol.FullyConnected(data = indata, weight = param$input.weight, bias = param$input.bias,
                                        num_hidden = num_hidden*5, name = paste0(prefix, "full", "_l", layeridx, "_s", seqidx))
  
  proj_full <- mx.symbol.split(proj_full, num_outputs = 5, axis=1)
  
  in_proj <- proj_full[[1]]
  mem <- mx.symbol.tanh(proj_full[[2]])
  write <- mx.symbol.tanh(proj_full[[3]])
  read <- mx.symbol.tanh(proj_full[[4]])
  highway <- mx.symbol.sigmoid(proj_full[[5]])
  
  if (is.null(prev.state)) {
    next.c <- write * mem
    next.h <- highway * in_proj
  } else {
    next.c <- prev.state$c + write * mem
    next.h <- read * prev.state$c + highway * in_proj
  }
  return(list(h = next.h, c = next.c))
}


# GRU cell symbol
gru.cell <- function(num_hidden, indata, prev.state, param, seqidx, layeridx, dropout = 0, prefix)
{
  if (dropout > 0 && layeridx > 1) 
    indata <- mx.symbol.Dropout(data = indata, p = dropout)
  
  i2h <- mx.symbol.FullyConnected(data = indata, weight = param$gates.i2h.weight, 
                                  bias = param$gates.i2h.bias, num_hidden = num_hidden * 2, 
                                  name = paste0(prefix, "t", seqidx, ".l", layeridx, ".gates.i2h"))
  
  if (!is.null(prev.state)) {
    h2h <- mx.symbol.FullyConnected(data = prev.state$h, weight = param$gates.h2h.weight, 
                                    bias = param$gates.h2h.bias, num_hidden = num_hidden * 2, 
                                    name = paste0(prefix, "t", seqidx, ".l", layeridx, ".gates.h2h"))
    gates <- i2h + h2h
  } else {
    gates <- i2h
  }
  
  split.gates <- mx.symbol.split(gates, num.outputs = 2, axis = 1, squeeze.axis = F, 
                                 name = paste0(prefix, "t", seqidx, ".l", layeridx, ".split"))
  
  update.gate <- mx.symbol.Activation(split.gates[[1]], act.type = "sigmoid")
  reset.gate <- mx.symbol.Activation(split.gates[[2]], act.type = "sigmoid")
  
  htrans.i2h <- mx.symbol.FullyConnected(data = indata, weight = param$trans.i2h.weight, 
                                         bias = param$trans.i2h.bias, num_hidden = num_hidden, 
                                         name = paste0(prefix, "t", seqidx, ".l", layeridx, ".trans.i2h"))
  
  if (is.null(prev.state)) {
    h.after.reset <- reset.gate * 0
  } else {
    h.after.reset <- prev.state$h * reset.gate
  }
  
  htrans.h2h <- mx.symbol.FullyConnected(data = h.after.reset, weight = param$trans.h2h.weight, 
                                         bias = param$trans.h2h.bias, num_hidden = num_hidden, 
                                         name = paste0(prefix, "t", seqidx, ".l", layeridx, ".trans.h2h"))
  
  h.trans <- htrans.i2h + htrans.h2h
  h.trans.active <- mx.symbol.Activation(h.trans, act.type = "tanh")
  
  if (is.null(prev.state)) {
    next.h <- update.gate * h.trans.active
  } else {
    next.h <- prev.state$h + update.gate * (h.trans.active - prev.state$h)
  }
  
  return(list(h = next.h))
}
