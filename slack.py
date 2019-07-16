import slackweb
import json
from collections import OrderedDict

slack_config_path = '/root/.slackconfig'

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def notify_start(experiment_name, config):
    slack_config = load_config(slack_config_path)
    slack = slackweb.Slack(url=slack_config['url'])
    username = slack_config['username']
    message = build_message(username, 'started !!', experiment_name, config)
    slack.notify(text=message)
    
def notify_finish(experiment_name, config, score):
    slack_config = load_config(slack_config_path)
    slack = slackweb.Slack(url=slack_config['url'])
    username = slack_config['username']
    message = build_message(username, 'finished !!', experiment_name, config)
    slack.notify(text=message)
    slack.notify(text=f'best score: {score}')
    
def notify_fail(experiment_name, config, error_name, error_message):
    slack_config = load_config(slack_config_path)
    slack = slackweb.Slack(url=slack_config['url'])
    username = slack_config['username']
    message = build_message(username, 'failed !!', experiment_name, config)
    slack.notify(text=message)
    slack.notify(text=f'{error_name}: {error_message}')
    
def build_message(username, message, experiment_name, config):
    no_send_list = ['device', 'df', 'train_index', 'valid_index', 
                    'azure_run', 'writer']
    send_config = OrderedDict()
    for k, v in OrderedDict(config).items():
        if k not in no_send_list:
            send_config[k] = v
    send_message = f'<@{username}> \n'\
                   f'experiment name: {experiment_name}\n'\
                   f'*{message}*\n'\
                   f'config:{json.dumps(send_config)}'
    return send_message
    