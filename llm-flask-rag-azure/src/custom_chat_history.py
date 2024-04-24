# from collections import deque
import redis
import os
from dotenv import load_dotenv, find_dotenv


# .env file is read locally with env vars
_ = load_dotenv(find_dotenv())

try:
    rhost=os.environ['REDIS_HOST']
    rport=int(os.environ['REDIS_PORT'])
    rpass=os.environ['REDIS_PASSWORD']
except KeyError:
    print (f"REDIS_HOST,REDIS_PORT,REDIS_PASSWORD dont  exist")
    qhost = ""
    qport = ""

if rpass == "None":
    rpass = None



# as we anyway start using OOP and create objects, why not to create a chat object instead of flask sessions?
class CustomChatBufferHistory(object):
    def __init__(self,
                 human_prefix_start='Human:', human_prefix_end='',
                 ai_prefix_start='AI:', ai_prefix_end=''):

        #redis
        self.redisClient = redis.StrictRedis(host=rhost, port=rport, db=0, password=rpass, decode_responses=True)
        #can be used as a unique id for each session of browser
        self.sessionId = "user1"
        self.memory = self.get_list_from_redis()



        self.human_prefix_start = human_prefix_start
        self.human_prefix_end = human_prefix_end
        self.ai_prefix_start = ai_prefix_start
        self.ai_prefix_end = ai_prefix_end
        

    def clean_memory(self):
        # self.memory.clear()
        self.redisClient.delete(self.sessionId)


    def get_list_from_redis(self, start=0, end=-1):
        try:
            # Retrieve list from Redis
            data_list = self.redisClient.lrange(self.sessionId, start, end)

            return data_list

        except Exception as e:
            print("Error:", e)
            return None
        
    def update_history(self, human_query, ai_response):
        try:
            # add item to list in Redis
            self.redisClient.rpush(self.sessionId, f'\n{self.human_prefix_start} {human_query} {self.human_prefix_end} ')
            self.redisClient.rpush(self.sessionId, f'\n{self.ai_prefix_start} {ai_response} {self.ai_prefix_end}')

            print("Item added to list on Redis successfully.")

        except Exception as e:
            print("Error:", e)