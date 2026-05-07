---
layout: default
title: "06 Attention and Transformers"
permalink: /vision/CS231N/06
#subtitle: 
use_math: true
parent: CS231N
grand_parent: vision
---

# 6. Attention and Transformers 

## RNNs
purpose
- handle sequential data. expanding types of problems NN can address

problem 
- one-to-many : inputting 1 image and ouputting a sequence of words (e.g. image captioning)
- many-to-one : inputting 1 sequence of frames and ouputting a classification 

→ so, Attention and Transformers 

## Attention and Transformers 
### core concepts
Attention Mechanisms
- new NN primitive that operates a sets of vectors 

Transformer Architecture
- NN with self-attention 

Ubiquity of Transformers
- dominant architecture for most large-scale, SOTA DL applications 
- image classification, image generation, text generation, audio processing ••• 

#### bottleneck problem in RNNs 
problem definition 
- English → Italian : lengths and word orders may differ
communication bottleneck
- the only info passed from the input → output : only through the fixed-length context vector
- if the long input - difficult to summarize all necessary info into a single vector 
- prevents the model from scaling effectively to longer texts or complex data 

#### solution 
→ let's look back at the entire input sequence at each step of generating the output 

---

### Attention 

#### Enhancing the Decoder with Attn. 
- encoder remains the same 
- decoder initialization : initial hidden state for the decoder 
- compute **alignment scores**
	- for each decoder h.s. - calculate scalar alignment score by comparing it with each encoder h.s.
	- indicate how much the decoder state matches each input token 
	- F_AT (simple linear layer) can compute the score by concatenating the decoder state & encoder state 
- softmax for attention weights 
	- apply softmax function to the **alignment scores** to obtain attention weights
	- form a probability distribution over the input tokens 
	- represents the model's focus on different parts of the input 
- compute the Context Vector $C_1$
	- new context vector $C_1$ : linear combination of the encoder h.s.
	- summarizes relevant input information for the current output step 
- Decoder RNN update 
	- previous output token $Y_{t-1}$ , previous h.s. $S_{t-1}$, newly computed context vector $C_1$
	- produce the next h.s. and the next output token 

- entire attention mechanism is : differentiable 
- the context vector attends to different parts of the input sequence 


#### Iterative Attn. Process
- repeat the process - repeated for every time step of the decoder RNN 
- for second time step $S_2$, a new context vector $C_2$ is computed by 
	- comparing the new decoder state $S_1$ with all encoder h.s.
	- calculating new **alignment scores** , apply softmax , get updated attention weights 

#### Attention Weight
insights into which parts of the input sequence the model focuses on when generating output, aiding interpretability 
- interpretability : introspection into the model's processing - revealing what it 'looks at' during task execution 
- visualization method : attn weights can be **visualized as a 2D grid** (probability distribution over the input sequence for each output word)


---

### Attention as a Computational Primitive

#### Attention operator 
- data vectors : info to be summarized (e.g. encoder hs)
- query vectors : query the data vector and produce output (e.g. decoder hs)
- atttention operator : take query vector, compares it with data vector, summarize data vector into a context vector, and produce the output vector 


#### Scaled Dot-Product Attention
- similarity function : compare query and data vectors 
	- simplification to dot product (simplest similarity function)
- scaling for stability
	- prevent vanishing gradient ( <- by interaction between dot products and softmax in high dimensions)
	- ensures better gradient flow
	- more scalable to higher-dimensional vectors 


#### Key and Value 
each data vector is projected into 2 distinct vectors (key & value)
- key vectors : compared with query vectors to compute alignment score 
- value vectors : compute the weighted sum for the output 

#### Cross-Attn and Self-Attn 
- cross-attn layer 
	- takes query vector and data vector 
	- comparing 2 different types of data (e.g. input vs. output in translation)
- self-attn layer
	- operates on a single set of input vectors 
	- each input vector - query, key, value 
	- used when processing a single type of data (e.g. comparing parts of an image with itself for image classification)
	- reuse the same computational primitives for different problem structure 


#### Masked Self-Attn and Multi-Head Attn

Masked Self-Attn
- control which inputs can attend to which others 
- useful for tasks like language modeling (model should not cheat by looking ahead in the sequence)


Multihead Self-Attn
- runs multiple independent copies of the self-attn in parallel
- purpose 
	- increases the model's capacity
	- allows it to learn different types of relationships simultaneously 



---

### Comparison 

#### RNN 
- operate 1D ordered sequence 
- limitation
	- fundamentally sequential and not parallelizable across the sequence - make them difficult to scale 

#### CNN
- operate multi-dimensional grids
- mix info. locally 
- high parallelizable 
- limitation 
	- struggle to build large RF without many layers or large kernels 

#### Self-Attn
- operate sets of vectors 
- generalizes to long sequences, no bottleneck, high parallelizable
- limitation
	- computationally expensive - more GPUs needed 



---

### Transformer 
self-attn layer - residual connection - layer normalization - Feedforward Network / Multi-layer perceptron (MLP)

