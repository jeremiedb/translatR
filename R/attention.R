### Attention mechanisms

# takes a score calculated from the attention mechanism and returns a ctx vector
get_ctx <- function(value, score, length = NULL) {
  
  if (!is.null(length)) score <- mx.symbol.SequenceMask(data=score, axis=1, use_sequence_length = T, sequence_length=length, value=-99999999)
  # [1, seq, batch]
  attn_wgt <- mx.symbol.softmax(score, axis=1)
  # [features x seq x batch] [1 x seq x batch] dot  -> [1, features, batch]
  ctx_vector <- mx.symbol.batch_dot(lhs = value, rhs = attn_wgt, transpose_a = T)
  # [1, features, batch] -> [features, batch]
  ctx_vector <- mx.symbol.reshape(ctx_vector, shape = c(0, 0))
  
  return(ctx_vector)
}

# bilinear attention
attn_bilinear <- function(encode, num_hidden) {
  
  init <- function() {
    # [features, seq, batch] -> [query_key_size, seq, batch]
    key <- mx.symbol.FullyConnected(data = encode$value, num_hidden = num_hidden, no_bias = T, flatten = F, name = "attn_key_FC")
    return(list(key = key, value = encode$value))
  }
  
  attend <- function(query, key, value, length = NULL, attn_init) {
    
    # [num_hidden, batch] -> [1, features, batch] 
    query <- mx.symbol.expand_dims(query, axis = 2)
    # [num_hidden x seq x batch] dot [1 x num_hidden x batch] -> [1 x seq x batch]
    score <- mx.symbol.batch_dot(lhs = key, rhs = query)
    ctx <- get_ctx(value = value, score = score, length = length)
    
    return(ctx)
  }
  return(list(init=init, attend=attend))
}


# dot-attention
attn_dot <- function(encode, query_key_size, scale = T, prefix) {
  
  init <- function() {
    # [features, seq, batch] -> [query_key_size, seq, batch]
    key <- mx.symbol.FullyConnected(data = encode$value, num_hidden = query_key_size, no_bias = T, flatten = F, name = "attn_key_FC")
    query_weight <- mx.symbol.Variable("query_weight")
    return(list(key = key, value = encode$value, length = encode$length, query_weight = query_weight))
  }
  
  attend <- function(query, key, value, length = NULL, attn_init) {
    
    # [features, batch] -> [query_key_size, batch]
    query <- mx.symbol.FullyConnected(data = query, num_hidden = query_key_size, weight = attn_init$query_weight, no_bias = T, flatten = F)
    if (scale) query <- query / sqrt(query_key_size)
    
    # [query_key_size, batch] -> [1, query_key_size, batch] 
    query <- mx.symbol.expand_dims(query, axis = 2)
    
    # [query_key_size x seq x batch] dot [1 x query_key_size x batch] -> [1 x seq x batch]
    score <- mx.symbol.batch_dot(lhs = key, rhs = query)
    
    # [query_key_size, batch]
    ctx <- get_ctx(value = value, score = score, length = length)
    
    return(ctx)
  }
  return(list(init = init, attend = attend))
}


# MLP-attention
attn_mlp <- function(encode, query_key_size, scale = T, prefix) {
  
  init <- function() {
    # [features, seq, batch] -> [query_key_size, seq, batch]
    key <- mx.symbol.FullyConnected(data = encode$value, num_hidden = query_key_size, no_bias = T, flatten = F, name = "attn_key_FC")
    query_weight <- mx.symbol.Variable("query_weight")
    mlp_weight <- mx.symbol.Variable("mlp_weight")
    return(list(key = key, value = encode$value, length = encode$length, query_weight = query_weight, mlp_weight = mlp_weight))
  }
  
  attend <- function(query, key, value, length = NULL, attn_init) {
    
    # [features, batch] -> [query_key_size, batch]
    query <- mx.symbol.FullyConnected(data = query, num_hidden = query_key_size, weight = attn_init$query_weight, no_bias = T, flatten = F)
    
    # [query_key_size, batch] -> [query_key_size, 1, batch] 
    query <- mx.symbol.expand_dims(query, axis = 1)
    
    # [query_key_size, 1, batch] -> [query_key_size, seq, batch]
    query <- mx.symbol.broadcast_add(lhs = key, rhs = query)
    if (scale) query <- query / sqrt(query_key_size)
    query <- mx.symbol.tanh(query)
    
    # [query_key_size x seq x batch] -> [1 x seq x batch]
    score <- mx.symbol.FullyConnected(data = query, num_hidden = 1, weight = attn_init$mlp_weight, no_bias = T, flatten = F)
    
    # [query_key_size, batch]
    ctx <- get_ctx(value = value, score = score, length = length)
    
    return(ctx)
  }
  return(list(init = init, attend = attend))
}
