translatR
================

Lightweight tools for translation tasks with MXNet R.

Inspiration taken from the [AWS Sockeye](https://github.com/awslabs/sockeye) project.

Getting started
---------------

Prepare data using WMT datasets using `Preproc_wmt15_train.Rmd`. Creates a source and target matrix of word indices along with the associated dictionary. Data preparation mainly relies on `data.table` package.

A RNN encoder-decoder training demo is provided in `NMT_rnn_rnn.Rmd` and and CNN-RNN architecture is shown in `NMT_cnn_rnn.Rmd`.

Performance
-----------

Performance during training is tracked using the perplexity metric. Once a training is completed, the above scripts show how to perform a batch inference on a test data, more specifically the WMT official one. [sacreBLEU](https://github.com/mjpost/sacreBLEU) is then used to compute the BLEU score, providing a clear comparison point to the metric typically found in publications.

For example:

`cat data/trans_test_wmt_en_fr_rnn_72_Capital.txt | sacrebleu -t wmt14 -l en-fr`

A BLEU score of 28.2 was obtained with the CNN-RNN architecture.

Features:
---------

#### Encoders:

-   Bidirectional RNN encoders (LSTM and GRU).
-   Convolutional encoder with residual gates.

#### Decoders:

-   RNN (LSTM and GRU)

#### Attention:

Architectures are fully attention based. All information is passed from the encoder to the decoder through the encoded sequences weighted with an attention mechanism (RNN hidden layers are not carry forward from the encoder to the decoder).

Attention modules are defined in `attention.R`. Three attention approaches are supported, all described in Luong et al, 2015:

-   Bilinear (`attn_bilinear`)
-   Dot (`attn_dot`)
-   MLP (`attn_MLP`)

The decoder network takes an attention module as a parameter along with an encoder graph. All attentions modules are implemented using the query-key-value approach.

To do:
======

-   Test multi-GPU support
-   More efficient data preprocessing to handle larger scale tasks (currently works fine up to around 4M parallel sequences).
-   Support for bucketing
-   Positional embedding
-   Transformer encoder-decoder
-   Encoder and decoder self-attention
-   Beam search

Tutorial to be added to [Examples of application of RNN](https://jeremiedb.github.io/mxnet_R_bucketing/index.html).
