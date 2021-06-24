# -Artificial Intelligent Project : Text Emotional Detection in Meme Chabot
## A. PROJECT SUMMARY

**Project Title:** Meme ChatBot

**Team Members:** 

![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/misc/Team%20Members.png)
<p align="center">
Figure 1 : Team Members Of The Project
</p>

**Objectives:**

- To create a chatbot that able to use meme in daily conversation as a form of entertainment for users.

## B. ABSTRACT
  A meme is an idea, behavior, or style that spreads by means of imitation from person to person within a culture and often carries symbolic meaning representing a particular phenomenon or theme. A meme acts as a unit for carrying cultural ideas, symbols, or practices, that can be transmitted from one mind to another through writing, speech, gestures, rituals, or other imitable phenomena with a mimicked theme.
  
  In this AI project, deep learning is used to train our chatbot to analyse and classify the user's emotion accurately which are 'Angry', 'Fear', 'Sad' , 'Love' , 'Surprise' ,and 'Joy'. When the user sent a message to the bot, the bot will choose randomly either reply a sentence only or a sentence with a meme image. If the bot is going to sent a meme, the bot will analyse the user's emotion and reply a suitable meme based on the user's emotion.
  
![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/misc/poster.png)
<p align="center">
Figure 2 : How The Chatbot Work
</p>
  
## C. DATASET
In this project, we’ll discuss about text emotional detection model, detailing on how our computer vision/deep learning pipeline will be implemented.

From there, we’ll review the dataset we’ll be using to train our text emotional detection model.

We’ll then show you how to implement a Python script to train a text emotional detection model using keras and Tensorflow.

We’ll use this Python script to train a text emotional detection model and review the results.

In order to train a meme chatbot, we need to break our project into three distinct phases, each with its own respective sub-steps :

- Preprocessing: We will focus on the pre-processing data of the development of the text emotional detection model

- Training: We will focus about how the model will detect the emotion from the input of user

- Deployment: We'll implement our text emotional detection model to meme chatbot by using discord bot as medium. Randomly picked meme will send to user based on user's text emotion when user chats with our chatbot.

![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/misc/phase.png)
<p align="center">
Figure 3 :  Phases And Individual Steps For Building A Meme Chatbot
</p>

For the dataset, we will categorised the emotion into 6 types, which is : Anger, Saddness, Joy, Love, Fear and Surprise.

![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/misc/sample%20data.PNG)
<p align="center">
Figure 4 : Dataset From Tweet Emotion Dataset Which Is From nlp Package
</p>
This dataset consists of 16000 sentences belonging to six classes:
- sadness = 4666 sentences
- anger = 2159 sentences
- love = 1304 sentences
- suprise = 572 sentences
- fear = 1937 sentences
- joy = 5362 sentences

Our distribution of dataset for training are shown as below:
<p align="center" width="100%">
  <img src="https://github.com/Choo99/Meme-Chatbot/blob/master/misc/dataset.png"><br>
  Figure 5 : Prepared Dataset Of Our Emotional Training
</p>



The dataset that we used is categorised into 6 which are 'Angry', 'Fear', 'Sad' , 'Love' , 'Surprise' ,and 'Joy'. Each category will contain related meme pictures towards the emotion.


## D. PROJECT STRUCTURE

The following directories are the structure of our project:
<p align="center" width="100%">
   <img src="https://github.com/Choo99/Meme-Chatbot/blob/master/misc/structure.PNG"><br>
  Figure 6 : 2 Directories Of The Project（Notebook And Python）
</p>
The notebook directory contains the record of our training process.

The python directory contains the Python scripts that we use in this project. We'll review three Python scripts:
- discordBot.py: This scripts will implement our trained text emotional detection model and our chatbot into a discord bot.
- emotional_detection.py: This scripts will using the trained model to detect the text emotion of user's input.
- emotional_training.py: Accepts our input dataset and use it to create our emotional_model. A training history containing accuracy/ loss curves is also produced
- meme directory: A well categorised memes set by emotion, which is our dataset of memes when chatbot need to send a meme to users.


## E. TRAINING
We are now ready to train our text emotional detection model using Keras, TensorFlow, and Deep Learning.

![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/misc/training.PNG)
<p align="center">
Figure 7 : Result Of Training And Validation
</p>
  
From the picture above, we are training our model for 500 steps and validate on 2000 samples in one epoch.
As a result, our model are able to get 97.66% of training accuracy and 88.20% of validation accurancy

The following is the training result of our model towards epochs of our training
![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/misc/graph.png)

Our test data for training are shown as below:
<p align="center" width="100%">
  <img src="https://github.com/Choo99/Meme-Chatbot/blob/master/misc/test%20data.PNG">
</p>
<p align="center">
Figure 8 : Prepared Test Data  
</p>
As you can see, we are obtaining a model with high accuracy in detecting our text emotion. It has predicted the correct result perfectly! Hope it wille achieve the same accuracy in our chatbot.


## F.  RESULT AND CONCLUSION
Now, our model is ready to detect emotion from text correctly. Let's implement it in our discord bot.
Thanks to prsaw in providing a powerful package of chatbot. 

Launch discord bot according to the following instruction:
1. Download all files in the github.
2. Open src/python/discordBot.py
3. Edit variable 'bot_token' in line 8 with your discord bot token.
4. Open command prompt, change directory to src/python and run command 'python discordBot.py'.
5. Invite your bot to your discord server and enjoy it!


## G.  PRESENTATION
[![demo](https://img.youtube.com/vi/KBIYCv7_xrE/0.jpg)](https://youtu.be/KBIYCv7_xrE)

## H. ACKNOWLEDGEMENT
* [Tweet emotional detection](https://www.coursera.org/projects/tweet-emotion-tensorflow)
* [Chatbot package](https://pypi.org/project/prsaw/)
* [Discord bot tutorial](https://realpython.com/how-to-make-a-discord-bot-python/)
