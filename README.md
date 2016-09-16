# char2vec

This code implements the skip-grap algorithm to find vector representations for the letters of the alphabet, as opposed to words as is done in word2vec.
It does this by taking a body of text (stored in `/data`) and training a shallow neural network to predict characters c_(n-1) and c_(n+1) given c_n. In this implementation, c_n is represented as a one-hot encoding, mapped to a hidden layer, and then mapped to two output layers (one each for c_(n-1) and c_(n+1)), with categorical cross-entropy losses.

The result this algorithm has is that characters which appear in similar contexts will have similar encodings. For example, vowels often appear in similar contexts, so we would expect them to have similar encodings.  

## Requirements

This code is written in Python and requires [Keras](https://keras.io/).

## Usage

```bash
$ python main.py
```

When the code is run, it will convert the entire text file to training data (watch out for RAM usage) and then train the model. Since the number of classes is quite small, the network should converge quite quickly. Next, the encodings for the characters will be generated and plotted.

## Additional Notes

The hidden layer/encoding is currently 2-D. This makes it easier to visualize without having to use techniques such as PCA or t-SNE.

The code currently uses window sizes of width 3 (c_(n-1:n+1)). There are several lines commented out which allow this width to be increased.

I have found that the text source can result in slightly different embeddings. Though for the same body of text, the embeddings learned between trials are very similar, up to rotation and flipping.

Something fun to try: instead of using a `tanh` activation in the hidden layer, use `softmax` with an encoding dimension << #chars -- this should allow you to come up with approximate classifications of the letters of the alphabet. This could also be achieved with clustering and the tanh activation... but this alternative approach seems more fun.
