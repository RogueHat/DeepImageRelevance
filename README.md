# DeepImageRelevance
A better way to learn about images.

*readme still under construction*

## Requirements
* Python >=3.6.9
* (Py)Torch >=1.4.0
* Torchvision
* SciPy
* Pillow
* dhash
* Danbooru 2019 512px dataset

## How to use
1. Execute create_directories.py
2. Download the 512px danbooru dataset into the danbooru folder
3. Go into the ml directory and execute train.py to generate the model. Runtime: A couple of minutes
4. Go into the index directory and execute create_indexes.py. Runtime: A really long time. About 2 days on my machine.
5. Go into the search directory and execute run_search.py. This will run a search against the test image and populate the results folder with relevant images. Runtime: 15 seconds

## Who
RogueHat - A software developer and novice machine learning hobbyist.

## When
Back in 2016, pytorch and a couple other machine learning frameworks became public and took the world by storm. At the time I had a large amount of unsorted fanmade artworks that I wanted to sort through. I had just learned about image categorization and thought this was a great solution; all I had to do was slice off the last layer and use transfer learning against a new set of tags, but then I ran into a problem...

## Why
Every single day, new fanmade and offical works are made from new titles. New tags are being created for every conceivable situation, from backgrounds, to number of persons in an image, even crossovers between titles. By the time I finished training a model, it would be behind in some way in terms of what works it can search or what the model even knows to search by. There had to be a better way.

## What
Enter Deep Image Relevancy: A model that can finds relevant images for a given image directly, *without using any metadata.*

## How
Alright that was a bit of an exaggeration, there is some metadata, but it was never used as part of training this model. Rather I am taking advantage of transfer learning to speed up the training process.

Here's the workflow:
1. Use an existing image categorization model and slice off the last layer to extract an image's features. (In this case: Mobilenet v2)
2. Train a variatonal autoencoder againsts those image's features
3. Create a database that uses the outputs of the VAE as indexes
4. Process a target image through the same model and use cosine similairty to find the top 20 images.

In essence, the existing image categorization model would interpret the image, while the VAE would try to understand the image and place it in context.

## How tho
That sounds impossible, you might say. How can a machine contextualize an image without involving metadata? The key is in the choice of the second model used: the Variational Autoencoder. 

Lets examine the existing image categorization model. The outputs from the model are specifically trained from an existing set of tags, which means that the space of images it is trying to describe is only defined by those set of tags. For example, a model that is trained to distinguish between a face with eyeglasses and a face with sunglasses will have no knowledge of a face with hats. Intuitively, we can guess that the hats will fall in between these two tags, but because the space of images is trained specifically in those tags, the model's behavior will not be defined in that space. i.e. its not an educated guess.

A variational autoencoder allieviates this problem by outputing not a coordinate, but rather a probability distribution in that space. By sampling and training against probability distributions, if there are any correlations between hats and either sun or eyeglasses, the space where hats would sit will still be defined as a result of those distributions overlapping. In essense, we learned about hats implicilty without ever introducing hats as a tag. Another result of this is that this approach will attempt to make any point in that space meaningful, even if we don't know about it yet.

Taken to the extreme, this concept can even be used to relate one set of tags to another set of tags, or more to the point, attempt to make every point in the latent space of images meaningful (as with the hats), and therefore searchable. With this in mind, you can find not just similar looking images, but *relevant* images as well.

## So what does this all mean
Originally this was meant as a way for me to retrain a different categorization model without having to retrain directly against the dataset, resulting in faster training times. Goal achieved I suppose. But in doing so I found (not discovered. I am sure people way smarter than me have already found this out) a way to represent images in a latent space that is meaningful and can be used in other applications. Image Relevancy was the first direct result of this. Knowing how VAE's work, I am sure someone can come along and use this to generate content as part of a GAN. Someone could throw an LSTM to the output of this thing and even map dialogue to that latent space. Kinda exciting not gonna lie, but in any case hope someone out there will find this useful.

## Implementation details
Unforunately my machine is way too underpowered to do the whole search on the VAE output alone. In order to speed up my search, I rely on a degraded version of dhash (a well known algorithm for image similarity) and run the relevancy search on the top 10000 images. Perhaps in the future using K Means as a smarter way of clustering the indexes. Theory is this clustering would actually train fast due to the nature of latent spaces being meaningful in Variational Auto Encoders.
