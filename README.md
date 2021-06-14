# -Artificial Intelligent Project : Meme Chabot
## A. PROJECT SUMMARY

**Project Title:** Meme ChabtBot

**Team Members:** 

![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/Team%20Members.png)
Figure 1 shows the team member of the project

**Objectives:**

- To create a chatbot that able to use meme in daily conversation as a form of entertainment for users.

## B. ABSTRACT
  A meme is an idea, behavior, or style that spreads by means of imitation from person to person within a culture and often carries symbolic meaning representing a particular phenomenon or theme. A meme acts as a unit for carrying cultural ideas, symbols, or practices, that can be transmitted from one mind to another through writing, speech, gestures, rituals, or other imitable phenomena with a mimicked theme.
  
  In this AI project, deep learning is used to train our chatbot to analyse and classify the user's emotion accurately which are 'Angry', 'Fear', 'Sad' , 'Love' , 'Surprise' ,and 'Joy'. When the user sent a message to the bot, the bot will choose randomly either reply a sentence only or a sentence with a meme image. If the bot is going to sent a meme, the bot will analyse the user's emotion and reply a suitable meme based on the user's emotion.
  
![Coding](https://www.todaysparent.com/wp-content/uploads/2017/06/when-your-kid-becomes-a-meme.jpg)
Figure 2 shows the Meme about the working hours

## C. DATASET
In this project, we’ll discuss about meme chatbot, detailing how our computer vision/deep learning pipeline will be implemented.

From there, we’ll review the dataset we’ll be using to train our meme chatbot.

I’ll then show you how to implement a Python script to train a meme chatbot in Discord Application.

We’ll use this Python script to train a meme chatbot and review the results.

There is an example of the meme chatbot will be looked like as shown in Figure 2:

In order to train a meme chatbot, we need to break our project into three distinct phases, each with its own respective sub-steps :

- Preprocessing: we will focus about the pre-processing data of the development of the chatbot

- Training: We will focus about how the chatbot will detect the key word or sentence of the input from user and reply them with some funny sentence 

- Deployment:After the chatbot is trained to reply, we will focus on the design of the meme chatbot like the colour and some icon design.


For the dataset, we'll be using an emotional training to train our chatbot to detect the emotional input of the input. There are 6 types of emotion will be categorised which is : Anger, Saddness, Joy, Love, Fear and Surprise.

Our dataset for training are shown as below:

![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/dataset.png)

Figure 3 shows the prepared dataset of our emotional training.
The dataset that we used is categorised into 6 which are 'Angry', 'Fear', 'Sad' , 'Love' , 'Surprise' ,and 'Joy'. Each category contain related meme pictures towards the emotion. 

Our test data for training are shown as below:

![Coding](https://github.com/Choo99/Meme-Chatbot/blob/master/test%20data.PNG)

Figure 4 shows the dataset prepared by oursleves.
The chatbot should detected the emotion in the sentence from user's input and reply. If the chatbot detected that the emotion of sentence typed by user is sad, the chatbot will reply with words and releted meme pictures which is about sad. 


Our goal is to train a meme chatbot to detect the emotional in the sentence of the input from a user and reply it with funny picture.

## D. PROJECT STRUCTURE

The following directory is our structure of our project:


## E. TRAINING
To train our chatbot, firstly we redirect our chatbot to the google drive folders which contain all the meme photo.



## F.  RESULT AND CONCLUSION


## G.  PRESENTATION

## H.  GUIDE TO RUN
1. Download all files in the github.
2. Open python/discordBot.py
3. Edit variable 'bot_token' in line 7 with your discord bot token.
4. Open command prompt, change directory to python run command 'python discordBot.py'.
5. Invide your bot to your discord server and enjoy it!
