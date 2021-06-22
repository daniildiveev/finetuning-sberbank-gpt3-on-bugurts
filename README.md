# Generating 'bugurts' using finetuned sberbank gpt-3

## Creating telegram application and config file

To create Telegram application you need an account. Since you are registered, complete the steps below.

1. Create telegram app [here](https://my.telegram.org/auth?to=apps).

2. Note down App api_id and App api_hash.

3. Join telegram group which information is needed. You can join telegram via group’s share link.

![](https://miro.medium.com/max/1400/1*TbQS21z5HkGY_tMd7CpbTw.png)

4. Now create `config.ini` file. Open and edit it, so it look something like that. 
 
NOTE! Creating a separate configuration file is optional! You can use regular variables, but using file is better in security purposes.
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
After that, create a Telegram client with respect to `username`, `api_id` and `api_hash`.
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
            #removing unnecessary  things
                text[0] = text[0].replace('БУГУРТ-ТРЕД', '')
                if ('https' in text[i]) or text[i] == '#БТnews': 
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
The `main()` fucntion. Again, note that it is used with `async` statement.

```Python
async def main(): 
    url  = 'https://t.me/bugurtthread' # Telegram channel url
    channel = await client.get_entity(url)
    messages = await get_all_messages(channel)
   
    #saving data to txt file
    with open('bugurts.txt', 'w', encoding='utf-8') as outfile:
        for i in range(len(messages)):
            outfile.write(messages[i])

async with client:
    await main()
```
After we've collected data, we dump it into txt file to save it. There are about 32000 posts parsed in a few minutes.

## Preparing model

I am using [Sberbank's gpt-3 (medium)](https://huggingface.co/sberbank-ai/rugpt3medium_based_on_gpt2), since it's russian and easy to finetune.

I load model using [Hugging Face repository](https://huggingface.co). 

Firstly, lets download `transformers` library using `pip`(if you use macOS use `pip3` instead) and load pretrained transformer model.

`datasets` library is used to create a dataset from our txt file.
```Python
#installing libs
!pip install transformers
!pip install datasets

from transformers import AutoTokenizer, AutoModelWithLMHead

#loading model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")

model = AutoModelWithLMHead.from_pretrained("sberbank-ai/rugpt3medium_based_on_gpt2")
```
Now we are ready to preprocess our data using `datasets` modules (`DataCollatorForLanguageModelling` and `TextDataset`)

Data collators are objects that will form a batch by using a list of dataset elements as input. 

`TextDataset` is used to convert txt file to dataset for finetuning our model.

```Python
from transformers import TextDataset, DataCollatorForLanguageModeling

train_path = 'drive/My Drive/bugurts.txt' #path to txt dataset file

def load_dataset(train_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, data_collator

train_dataset, data_collator = load_dataset(train_path,tokenizer)
```
If you did everything right, the output of `train_dataset[0]` should look similar to that:

![](https://github.com/sexozavr/finetuning-sberbank-gpt3-on-bugurts/blob/main/dataset_sample.png)

## Finetuning model

Okay, now we are ready for the main part: training our model.

For this action we are going to use `Trainer` and `TrainingArguments` modules.

```Python
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead

training_args = TrainingArguments(
    output_dir = "./gpt3-bugurts", #The output directory
    overwrite_output_dir = True, #overwrite the content of the output directory
    num_train_epochs = 10, # number of trainig epochs
    per_device_train_batch_size = 8, # batch size for training
    save_steps = 7000, # after steps model is saved
    warmup_steps = 500,# number of warmup steps for learning rate scheduler
    prediction_loss_only = True,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)
```
You can use different params for both `Trainer` and `TrainingArguments`. Just remember transformer's big boy, dont forget about your ram and disk capabilities.

After these actions, just run `trainer.train()`.

You will get something like that:

Step|Training loss|
-----|-------------|
500	 |     2.719400|
1000 |     2.574300|
1500	|     2.518100|
2000	|     2.461900|
2500	|     2.461900|
3000	|     2.425500|
3500	|     2.402100|
4000	|     2.381200|
4500	|     2.202700|
5000	|     2.203200|
5500	|     2.206700|
6000	|     2.203400|
.... |.........    |

That wraps it up for the finetuning, now lets see how our model performs.

## Generating with model

After you successfully finetuned model, you can generate text(bugurts).

Before generating, dont forget to switch your model to last checkpoint save.

That is done by running:
```Python
model = AutoModelWithLMHead.from_pretrained('/content/gpt3-bugurts/checkpoint-<your_last_checkpoint_step>')
```

To generate samples easily in one line i wrapped the whole generation process in the `model_generate` function.

```Python
def model_generate(model, start_token:str,repetition_penalty = 1.) -> str:
    start_token = start_token.upper()

    tokenized_text = tokenizer.encode(start_token, return_tensors='pt')
    greedy_output = model.generate(tokenized_text, 
                                max_length=200, 
                                do_sample=True, 
                                repetition_penalty=repetition_penalty)
    result = tokenizer.decode(greedy_output[0])
    result = result.split('@')

    return '@\n'.join([result[i] + ' \n' for i in range(len(result))])
```

To generate text simply pass your model, start token (the text you want model to continue) and repeptition penalty argument (this parameter will penalty model with certain value if model generates something that already had been generated) to `model_generate` function.

NOTE! Be careful to play with `repetition_penalty` parameter, if you are scared, just set it to 1.0. 

## Saving model

If you are interested in your sanity and your time managment, you better save your model, since the whole finetuning process takes a few hours.

To do this, simply run the following command:

```Python
model.save_pretrained('drive/My Drive/bugurt_gpt-3/', save_config=True, save_function=torch.save)
```

That's it!
