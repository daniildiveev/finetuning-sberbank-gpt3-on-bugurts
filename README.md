# Generating 'bugurts' using finetuned sberbank gpt-3

## Creating telegram application and config file

To create Telegram application you need an account. Since you are registered, complete the steps below.

1. Create telegram app [here](https://my.telegram.org/auth?to=apps).

2. Note down App api_id and App api_hash.

3. Join telegram group which information is needed. You can join telegram via group’s share link.

![](https://miro.medium.com/max/1400/1*TbQS21z5HkGY_tMd7CpbTw.png)

4. Now create `config.ini` file. Open and edit it, so it look something like that. 
```Python
[Telegram]
# no need for quotes

# you can get telegram development credentials in telegram API Development Tools
api_id = Telegram-API-ID
api_hash = Telegram-API-Hash

# use full phone number including + and country code
phone = Your-Telegram-Phone-Number
username = Your-Telegram-Username
```
Congrats! You've created Telegram app!

## Collecting and preprocessing data

So, we need to collect thе training data.
To do this, we have to parse some posts from Telegram channel. I decided to do this using Telethon library, so firstly lets setup Telegram client. 

```Python
#downloading modules
!pip install telethon
!pip install configparser
```
NOTE! If you are a macOS user, use `pip3` instead of `pip`.

`configparser` module is required to get our telegram application data(api_id, api_hash etc.)
```Python
import configparser
import json
from telethon import TelegramClient
```
Now lets open our `config.ini` by using `configparser` and get your login data.

```Python
# reading login data
config = configparser.ConfigParser()
config_path = 'config.ini'
config.read(config_path)

# getting settings from config.ini
api_id   = config['Telegram']['api_id']
api_hash = config['Telegram']['api_hash']
username = config['Telegram']['username']
```
After that, create a Telegram client with respect to `api_id` and `api_hash`.
```Python
client = TelegramClient(username, api_id, api_hash)

client.start()
```
Okay, now we are ready to parse posts.

NOTE! `telethon` is an asynchronous library, so everything will work only by using `async` statement.


This is the function we'll use to collect data from telegram channel.
```Python
async def get_all_messages(channel):
    '''Function to collect all "bugurts"'''
    offset_msg = 0    # index of post to start with
    limit_msg = 1000   # maximum number of messages to pass at once

    all_messages = []   #all messages list
    total_messages = 0
    total_count_limit = 0  # change this value if you dont need all of the messages
    
    while True:
        history = await client(GetHistoryRequest(
            peer=channel,
            offset_id=offset_msg,
            offset_date=None, add_offset=0,
            limit=limit_msg, max_id=0, min_id=0,
            hash=0))
        if not history.messages:
            break
        messages = history.messages
        for message in messages:
            if 'message' not in message.to_dict().keys(): continue
            text = message.to_dict()['message']# the message is a dict with a lot of params, the message content is in the 'message'
            text = text.split('\n')
            for i in range(len(text)):
            '''removing unnecessary  things
                text[0] = text[0].replace('БУГУРТ-ТРЕД', '')
                if ('.ru' in text[i]) or text[i] == '#БТnews': 
                    text[i] = ''
            text = ' '.join(text)
            all_messages.append(text)
        offset_msg = messages[len(messages) - 1].id
        total_messages = len(all_messages)
        print(f'{total_messages} / ± 32000 posts')
        if total_count_limit != 0 and total_messages >= total_count_limit:
            break
    return all_messages
```
The `main()` fucntion, again, note that it is used with `async` statement




















