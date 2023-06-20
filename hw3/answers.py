r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=512,
        seq_len=32,
        h_dim=256,
        n_layers=3,
        dropout=0.1,
        learn_rate=1e-4,
        lr_sched_factor=0.2,
        lr_sched_patience=2,
    )
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======

    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**

First of all, our model is designed to handle sequential data - meaning the model will use the last input while
while processing the recent input. By splitting the corpus into sentences we allow the model to process
a sentence at a time, while using the last sentence to aid in the process. if we wouldn't split the corpus we will not be 
able to process the data sequentially.
Secondly, the corput might be very large requiring large computational resources, splitting the corpus and loading only 
a part of the corpus every time is more memory efficient.
Lastly, is might also aid with exploding/vanishing gradients, as using the chain rule with a large amounts of gradients
in the backpropagation process might result in exploding/vanishing gradients.
"""

part1_q2 = r"""
**Your answer:**

When generating text, the model takes into consideration the previous input processed by the model,
thus, the last proccesed input stays in memory in order to use it when processing the next input- meaning 
we indeed maintain in memory more data then the last sequence only, 

"""

part1_q3 = r"""
**Your answer:**

We dont shuffle the batch order while training because the sentences order is important for a logical 
flow of the text.
In other words, we created a spesial sampler called "SequenceBatchSampler" so each sample at index i in batch j is the 
continuation of the sample sample i at batch j-1. By that we maintain the logical flow of the text.
If we would shuffle the batched we would ruin the logical flow and might also loss the context, resulting in a less logical text.


"""

part1_q4 = r"""
**Your answer:**


"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers = dict(
        batch_size=8,
        h_dim=1024,
        z_dim=64,
        x_sigma2=1e-2,
        learn_rate=2e-4,
        betas=(0.5, 0.998),
    )
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

The $\sigma^2$ hyperparameter in the VAE loss function basically controls the trade-off between the two terms:
data-reconstruction and KL-divergence.
The data-reconstruction term tries to minimize the difference between the reconstructed data and the original data,
And the KL-divergence term tries to minimize the difference between the approximate posterior distribution of the latent
variables and prior distribution. Thus, the $\sigma^2$ hyperparameter basically acts as a regularization *coefficient*

The effect of a low value of $\sigma^2$, is that the KL-divergence term will have a small weight when minimizing the 
loss function, resulting in a model that has a lower ability of generating new samples (as we didn't focus on difference
between the *approximate posterior and prior distribution), but has a higher ability of reconstructing the original 
data accurately.
In the extreme case where $\sigma^2$ -> 0, the model is a basic auto-encoder.

The effect og high values of $\sigma^2$, is that the data-reconstruction term will have a small weight when minimizing
the loss function, resulting in a model that has a higher ability to generate new data (as we focused on minimizing the
distance between approximate posterior and prior distribution), thus, the model is able to generate new samples that are
similar to the original data, but has lower abilities of reconstructing accurately the original data samples. 
"""

part2_q2 = r"""
**Your answer:**

***1)***    As we learned the VAE lose is composed of two terms: data-reconstruction and KL-divergence.
The data-reconstruction measures the distance between reconstructed and the original data- meaning it measures the
decoder's ability to reconstruct the original data from it's latent representation.
The KL-divergence measures the difference of the approximate posterior and prior distributions- meaning it measures the 
ability to generate new samples that are similar to the original ones. This term can be seen as a regularization term,
as minimizing it, will in return minimize the difference between the approximate posterior and prior distributions
which pushes the latent representation to represent closely the original data, thus, aiding in generating similar samples.
 
***2)***     The KL-divergence loss term affects the latent-space distribution, since the KL-divergence term measures the
dis-similarity between the latent-space and prior distributions, minimizing the term enforces the latent-distribution to 
be similar to the prior distribution. 
Thus, the latent-space variables will have a similar distribution as the prior ones.
As we learned, the prior distribution is usually a simple one, for example a
normal distribution. 
Moreover, this can be seen as some sort of generalization, since the latent variables will be distributed in a in a
similar way to the prior (as explained usually a normal distribution) thus making the model more generalized and informative. 
 
***3)***    There are multiple benefits of this effect:

A)  Improving generalization and avoiding overfitting: as explained, by regularizing the latent-space, the VAE is 
encouraged to produce latent variables that are distributed similarly to the prior distribution making it harder for the
model to overfit to the training data. Moreover, we stated that the prior distribution is a simple one, thus the 
latent-space is in return similar to this simple distribution making it generalize better to unseen data. 
Lastly, since this simple distribution is ussually a normal one, when we generate new samples from the model, we basically
sample them from a normal distribution.

B)  Trade-off between the data-reconstruction loss and KL-divergence loss: As explained above, the KL-divergence term 
controls the trade-off between the two losses, resulting in a potentially balanced loss, since the first term refers to 
the ability to reconstruct the  original data, and the second to the ability to genarate new similar samples - meaning a
balanced trade-off will result in an overall better model.

C)  Simple latent-space distribution: since the prior distribution is simple and the latent-space distribution is 
pushed to be similar to the prior one, we get a simple distribution in the latent space, which has benefits as we
addressed above.
"""

part2_q3 = r"""
**Your answer:**



"""

part2_q4 = r"""
**Your answer:**

We model the log of the variance instead of directly modeling the variance for multiple reasons:
First of all, it is important to note that this is a common practice in statistics when trying to maximize a likelihood 
function. It is preferred because it is usually easier to analyze the log likelihood, and that is at all possible because
the log function is monotone.
Secondly, using the log of the variance enhances the numerical stability- since we are modeling $\bb{\sigma}^2_{\bb{\alpha}}$,
there is a constraint that the variance must be non-negative (power of 2), because optimization algorithms allow the 
parameters to receive any real value, we could encounter a problem where the variance takes a negative number or converge to zero.
For that reason, optimizing the log-sacle is more stable and reliable then optimizing the variance directly because the
log function maps positive values to the entire real number line, avoiding potential problems with negative of zero 
values, thus ensuring that the variance will always be a positive number.


"""

# ==============

# ==============
# Part 3 answers

PART3_CUSTOM_DATA_URL = None


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


# ==============
