## **Character Level Machine Translation**
This project, originally an evaluation component for the Deep Learning course (2022/2023), talking place in Instituto Superior Técnico, University of Lisbon, aimed to explore the concept of **machine translation - the problem of automatically translating a sequence of tokens from a source language into a target language**. More specifically, this project contains a **character-level machine translation model from Spanish to English**.

REPLACE
<p align="center">
  <img src="https://github.com/user-attachments/assets/598e0554-0dff-4928-82bc-ea1ffdb41e92"/>
</p>

<p align="center">
  <i>Figure 1 - Images from the Kuzushiji-MNIST dataset</i>
</p>

The following document indicates how to access and utilise the source code. It also contains a brief analysis of the implementation and results.

## **Quick Start**
This project's source files can be downloaded from this repository. They are divided into the following main files:
- ***hw2-q3*** - contains a Perceptron, a logistic regression and a multi-layer Perceptron classifiers, all grounded on "manual" implementations of differentiation.

To run this poject, follow these steps:
1. Install the necessary dependencies:
     - pip install torch
     - pip install scikit-learn
     - pip install matplotlib
     - pip install torchvision
  
2. Simply run whatever file you would like utilising a terminal. Bare in mind some models accept input parameters. Examples:
     - python hw1-q1.py logistic_regression
     - python hw1-q2.py ffn -layers 1 -hidden_sizes 200
     - python hw1-q2.py -layers 2 -hidden_sizes 200 -learning_rate 0.1
  
Feel free to change the test and training sets, as well as any other parameters you see fit.

## **Initial Implementation - Without Attention Mechanism**
A **character-level machine translation model from Spanish to English** was implemented (*python hw2-q3.py*) considering the **mean error rate** (which calculates the Levenshtein distance between the prediction and the true target, divided by the true length of the sequence) **as the main evaluation metric**. The model is **grounded on an encoder-decoder architecture with an autoregressive LSTM as the decoder and a bidirectional LSTM in the encoder**. 

The underlying dataset was provided before the project and is inside the *data* folder. The model was **trained over 50 epochs**, utilising the following **parameters**: 
- A learning rate of 0.003
- A dropout rate of 0.3
- A hidden size of 128
- A batch size of 64

The model's performance is displayed in Figure 1, having achieved a final validation error rate of 0.4847 and a final test error rate of 0.4906.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e632f733-2b80-4b28-97eb-a32c5e54b7ab"/>
</p>

<p align="center">
  <i>Figure 1 - Validation error rate over 50 epochs of Encoder-Decoder model with Bidirectional LSTM and Auto-regressive LSTM</i>
</p>

## **Advanced Implementation - With Attention Mechanism**
A **billinear attention mechanism was developed for the decoder**, which **weighs the contribution of the different source characters**, according to relevance for the current prediction. The previously employed hyperparameters were once again utilised. This version of the model can be executed with the command *python hw2-q3.py –use_attn*.

The model's performance is displayed in Figure 2, having achieved a final validation error rate of 0.3828 and a final test error rate of 0.3862, which represents a **clear improvement when compared to the previous version of the translator**.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8e6ffa23-44bd-4573-a69c-486d6f02bdcd"/>
</p>

<p align="center">
  <i>Figure 2 - Validation error rate over 50 epochs of Encoder-Decoder model with Bidirectional LSTM and Auto-regressive LSTM</i>
</p>

## **Future Improvements**
There are multiple possible ways of further improving the model’s performance without altering its encoder-decoder architecture, most of which are popular current topics of research. However, before mentioning these improvements, let us reflect on the optimizations it already has.

The **encoder-decoder model consisting of recurrent neural networks (RNN) suffers from shortcomings like the vanishing gradient, mainly due to its chronological architecture** - information belonging to the beginning of the sequence is often “lost” during training if the network is big enough and **does not have mechanisms to handle long-term dependencies**. 

The development of techniques like the **long short-term memory (LSTM)** help minimise this issue by implementing **gates (extra weights) that selectively forget or retain information from previous time steps**. The **attention mechanism** was also introduced as a means to solve this long-distance dependence problem by **reviewing and selecting the most important information in the whole input sequence**. Lastly, by simultaneously **capturing the “past” and “future” context of any given token**, **bidirectional encoders** also boost the results of machine translation algorithms.

In the current implementation of the model, the decoding process comes to a halt when the sequence reaches a maximum length of 50 characters or when the token marking the end of the sequence is reached. One way of **improving the model’s performance** would be to **increase the maximum possible length of the generated sequence (increasing the for-loop duration), thus giving more time to generate a correct translation**. In other words, this would increase the likelihood of the model reaching the final token with a complete and acceptable translation.

On the other hand, we could **apply a beam search technique** instead of utilising a single best candidate for each position of the generated sequence (greedy search) - multiple tokens could be considered based on conditional probability. This also means that rather than having just the highest-probability sequence at each step, the model would **store a list of multiple highest-probability sequences**. Although this could take a toll computationally wise, the quality of the results would undoubtedly increase because multiple translation alternatives would be explored.

An **early stop mechanism** could also be considered during training in order to **prevent overfitting and lack of generalisation capabilities**. This technique would **stop the training process once the accuracy of the model on a validation started decreasing**.

Finally, perhaps the most general improvement for a machine translation algorithm - **quite simply expanding the size and the diversity of the dataset** would develop the generalisation of the model thus yielding better results.

## **Authors and Acknowledgements**
This project was developed by **[Miguel Belbute (zorrocrisis)](https://github.com/zorrocrisis)** and [Guilherme Pereira](https://github.com/the-Kob).

The skeleton code was supplied by Francisco Melo (fmelo@inesc-id.pt).
