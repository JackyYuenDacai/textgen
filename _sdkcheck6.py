import urllib.request
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    r = urllib.request.urlopen(req, timeout=60)
    data = b''
    while True:
        chunk = r.read(65536)
        if not chunk:
            break
        data += chunk
    return data.decode('utf-8', 'ignore')

base = 'https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/responses/'
for name in ['response_queued_event.py', 'response_in_progress_event.py']:
    try:
        print('=====', name)
        print(fetch(base + name))
    except Exception as e:
        print(name, 'ERR', e)
