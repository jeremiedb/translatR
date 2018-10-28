### Attention mechanisms

attn_key_create <- function(encode, num_proj_key = NULL) {
  
  # assign value to encode
  value <- mx.symbol.identity(encode, name = "encode_value")
  
  if (!is.null(num_proj_key)) {
    attn_key_weight <- mx.symbol.Variable("attn_key_weight")
    key <- mx.symbol.FullyConnected(data = encode, num_hidden = num_proj_key, weight = attn_key_weight, no_bias = T, flatten = F, name = "key")
  } else key <- mx.symbol.identity(encode, name = "key")
  return(list(key = key, value = value))
}


# takes a score calculated from the attention mechanism and returns a ctx vector
get_ctx <- function(value, score, length = NULL) {
  
  if (!is.null(length)) score <- mx.symbol.SequenceMask(data=score, axis=1, use_sequence_length = F, sequence_length=length, value=-9999)
  # [1, seq, batch]
  attn_wgt <- mx.symbol.softmax(score, axis=1)
  # [features x seq x batch] [1 x seq x batch] dot  -> [1, features, batch]
  ctx_vector <- mx.symbol.batch_dot(lhs = value, rhs = attn_wgt, transpose_a = T)
  # [1, features, batch] -> [features, batch]
  ctx_vector <- mx.symbol.reshape(ctx_vector, shape = c(0, 0))
  
  # [1, seq, batch] -> # [seq, batch]
  # attn_wgt <- mx.symbol.reshape(data=attn_weight, shape = c(0, 0))
  
  return(ctx_vector)
}

# bilinear attention
attn_bilinear <- function(query, value, num_hidden) {
  
  init <- function() {
    # [features, seq, batch] -> [query_key_size, seq, batch]
    key <- mx.symbol.FullyConnected(data = value, num_hidden = num_hidden, weight = query_proj_weight, no_bias = T, flatten = F)
    query_weight <- mx.symbol.Variable("query_weight")
    return(list(key = key, value = value, query_weight = query_weight))
  }
  
  attend <- function(query, key, value, query_weight) {
    
    # [num_hidden, batch] -> [1, features, batch] 
    query <- mx.symbol.expand_dims(query, axis = 2)
    # [num_hidden x seq x batch] dot [1 x num_hidden x batch] -> [1 x seq x batch]
    score <- mx.symbol.batch_dot(lhs = key, rhs = query)
    ctx <- get_ctx(value = value, score = score, length = NULL)
    
    return(ctx)
  }
  return(list(init=init, attend=attend))
}


# dot-attention
attn_dot <- function(value, query_key_size, scale = T) {
  
  init <- function() {
    # [features, seq, batch] -> [query_key_size, seq, batch]
    key <- mx.symbol.FullyConnected(data = value, num_hidden = query_key_size, no_bias = T, flatten = F)
    query_weight <- mx.symbol.Variable("query_weight")
    return(list(key = key, value = value, query_weight = query_weight))
  }
  
  attend <- function(query, key, value, query_weight) {
    
    # [features, batch] -> [query_key_size, batch]
    query <- mx.symbol.FullyConnected(data = query, num_hidden = query_key_size, weight = query_weight, no_bias = T, flatten = F)
    if (scale) query <- query / sqrt(query_key_size)
    
    # [query_key_size, batch] -> [1, query_key_size, batch] 
    query <- mx.symbol.expand_dims(query, axis = 2)
    
    # [query_key_size x seq x batch] dot [1 x query_key_size x batch] -> [1 x seq x batch]
    score <- mx.symbol.batch_dot(lhs = key, rhs = query)
    
    # [query_key_size, batch]
    ctx <- get_ctx(value = value, score = score, length = NULL)
    
    return(ctx)
  }
  return(list(init = init, attend = attend))
}


attention_ini <- function(key, value) {
  
  ini_weighting_weight <- mx.symbol.Variable("ini_weighting_weight")
  score <- mx.symbol.FullyConnected(data = key, num_hidden = 1, weight = ini_weighting_weight, no_bias = T, flatten = F)
  
  # attention - softmax applied on seq_len axis
  attn_wgt <- mx.symbol.softmax(score, axis = 1)
  # ctx vector:  [1 x seq x batch] dot [features x seq x batch] -> [features x 1 x batch]
  ctx_vector <- mx.symbol.batch_dot(lhs = attn_wgt, rhs = value, transpose_a = T)
  ctx_vector <- mx.symbol.reshape(ctx_vector, shape = c(-1, 0))
  return(ctx_vector)
}
