import discord
import os
import random
from prsaw import RandomStuff
import emotional_detection 

client = discord.Client()
api_key = 'HdRKCmeVl8zV'
rs = RandomStuff(async_mode = True , api_key =api_key)

types = ['angry','fear','joy','love','sad','surprise']
memes = { 'angry':[],'fear':[],'joy':[],'love':[],'sad':[],'surprise':[] }

for type in types:
 for filename in os.listdir('meme/' + type):
  with open(os.path.join('meme',type,filename), 'rb') as f: # open in readonly mode
   memes[type].append(discord.File(f))

channel = []

def update_channelList(channel_name):
  if "channel" in db.key():
    channel = db["channel"]
    channel.append(channel_name)
    db[channel] = channel
  else:
    db["channel"] = [channel_name]

def delete_channelList(index):
  channel = db["channel"]
  if len(channel) > index:
    del channel[index]
    db["channel"] = channel

def memes_picker(message):
 emotion = emotional_detection.predicting(message)
 i = random.randint(0, len(memes[emotion]) - 1)
 response = memes[emotion][i]
 return response

@client.event
async def on_ready():
  print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):

  if message.author == client.user:
    return

  elif message.content == ">set":
    channel.append(str(message.channel))
    await message.reply(str(client.user) + ' has been added into ' + str(message.channel) + ' channel by ' + str(message.author)) 
    
  elif str(message.channel) in channel:
    if message.content.startswith('>joke'): 
      response = await rs.get_joke(_type = "any")
      reply = ''
      if 'joke' in response:
        reply += response['joke']
      elif 'setup' in response:
        reply += response['setup']
      if 'delivery' in response:
        reply += ' ' + response['delivery']
      await message.reply(reply)  
    
    elif message.content == 'joy':
     i = random.randint(0, len(memes['joy']) - 1)
     await message.reply(file = memes['joy'][i])

    elif message.content == ">image":
      response = await rs.get_image(_type = "dankmemes")
      await message.reply(response)  

    else:
     response = await rs.get_ai_response(message.content)
     i = random.randint(0, 1)
     if i == 0:
      await message.reply(response[0]['message'])
     else:
      await message.reply(response[0]['message'],file = memes_picker(message.content))
     

  
client.run('ODQxNTE3NjQzNTIwNjcxNzQ0.YJn6YA.HEfHJ0a9vFCFN4psm5AsuQH3EzM')

