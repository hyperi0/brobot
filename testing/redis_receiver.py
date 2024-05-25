import redis

r = redis.Redis(host='localhost', port=6379, db=0)

while True:
    message = r.brpop('sentiment_scores')
    if message:
        print('Received: ', message[1].decode('utf-8'))