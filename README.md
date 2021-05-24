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

























