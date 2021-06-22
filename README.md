# -Artificial Intelligent Project : Text Emotional Detection in Meme Chabot
## A. PROJECT SUMMARY

**Project Title:** Meme ChabtBot

**Team Members:** 

![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/misc/Team%20Members.png)

Figure 1 shows the team member of the project

**Objectives:**

- To create a chatbot that able to use meme in daily conversation as a form of entertainment for users.

## B. ABSTRACT
  A meme is an idea, behavior, or style that spreads by means of imitation from person to person within a culture and often carries symbolic meaning representing a particular phenomenon or theme. A meme acts as a unit for carrying cultural ideas, symbols, or practices, that can be transmitted from one mind to another through writing, speech, gestures, rituals, or other imitable phenomena with a mimicked theme.
  
  In this AI project, deep learning is used to train our chatbot to analyse and classify the user's emotion accurately which are 'Angry', 'Fear', 'Sad' , 'Love' , 'Surprise' ,and 'Joy'. When the user sent a message to the bot, the bot will choose randomly either reply a sentence only or a sentence with a meme image. If the bot is going to sent a meme, the bot will analyse the user's emotion and reply a suitable meme based on the user's emotion.
  
![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/misc/poster.png)

Figure 2 shows that how our chatbot work

## C. DATASET
In this project, we’ll discuss about text emotional detection model, detailing how our computer vision/deep learning pipeline will be implemented.

From there, we’ll review the dataset we’ll be using to train our text emotional detection model.

We’ll then show you how to implement a Python script to train a text emotional detection model using keras and Tensorflow.

We’ll use this Python script to train a text emotional detection model and review the results.

In order to train a meme chatbot, we need to break our project into three distinct phases, each with its own respective sub-steps :

- Preprocessing: we will focus about the pre-processing data of the development of the text emotional detection model

- Training: We will focus about how the model will detect the key word or sentence of the input from user

- Deployment: We'll implement our text emotional detection model to our discord bot with our chatbot. Randomly picked meme will send to user base on user text emotion when user chat with our chatbot.

![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/misc/phase.png)

Figure 3 is the phases and individual steps for building a meme chatbot


For the dataset, there are 6 types of emotion will be categorised which is : Anger, Saddness, Joy, Love, Fear and Surprise.

![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/misc/sample%20data.PNG)

Figure 4 shows the dataset from Tweet Emotion Dataset which is from nlp package

This dataset consists of 16000 images belonging to six classes:
- sadness: 4666 sentences
- anger = 2159 sentences
- love = 1304 sentences
- suprise = 572 sentences
- fear = 1937 sentences
- joy = 5362 sentences

Our distribution of dataset for training are shown as below:

![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/misc/dataset.png)

Figure 5 shows the prepared dataset of our emotional training.
The dataset that we used is categorised into 6 which are 'Angry', 'Fear', 'Sad' , 'Love' , 'Surprise' ,and 'Joy'. Each category will contain related meme pictures towards the emotion.


## D. PROJECT STRUCTURE

The following directory is our structure of our project:

![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/misc/structure.PNG)

Figure 6 shows the 2 directories of our project which are notebook and python

The notebook directory contains the record of our training process.

The python directory contains the Python scripts that we use in this project. We'll reviewing three Python scripts:
- discordBot.py: This scripts will implement our trained text emotional detection model and our chatbot into a discord bot.
- emotional_detection.py: This scripts will using the trained model to detect the text emotion of user's input.
- emotional_training.py: Accepts our input dataset and use it to create our emotional_model. A training history containing accuracy/ loss curves is also produced
- meme directory: A well categorised memes set by emotion, which is our dataset of memes when chatbot need to send a meme to users.


## E. TRAINING
We are now ready to train our text emotional detection model using Keras, TensorFlow, and Deep Learning.
![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/misc/training.PNG)

Figure 7 shows the result of training and validation

From the picture above, we are training our model for 500 steps and validate on 2000 samples in one epoch.
As a result, our model able to get an 97.66% of training accuracy and 88.20% of validation accurancy

The following is the training result of our model towards epochs of our training
![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/misc/graph.png)

Our test data for training are shown as below:

![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/misc/test%20data.PNG)

Figure 8 shows the test data prepared by oursleves.
As you can see, we are obtaining a model with high accurancy in detect our text emotion. It is predicted the correct result perfectly! Hope it will be archieve same high accurancy in our chatbot.


## F.  RESULT AND CONCLUSION
Now, our model is ready to detect emotion from text correctly. Let's implement it in our discord bot.
Thanks to prsaw in provide a powerful package of chatbot. 

Launch discord bot with following instruction:
1. Download all files in the github.
2. Open src/python/discordBot.py
3. Edit variable 'bot_token' in line 8 with your discord bot token.
4. Open command prompt, change directory to src/python and run command 'python discordBot.py'.
5. Invite your bot to your discord server and enjoy it!


## G.  PRESENTATION

## H. ACKNOWLEDGEMENT
* [Tweet emotional detection](https://www.coursera.org/projects/tweet-emotion-tensorflow)
* [Chatbot package](https://pypi.org/project/prsaw/)
* [Discord bot tutorial](https://realpython.com/how-to-make-a-discord-bot-python/)