import urllib.request, re
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

base = 'https://raw.githubusercontent.com/openai/openai-python/main/src/openai'
try:
    html = fetch(base + '/types/responses/response_stream_event.py')
    print('len', len(html))
    events = sorted(set(re.findall(r'"(response\.[a-z_]+\.[a-z_]+)"', html)))
    print('event types in SDK union:')
    for e in events:
        print(' ', e)
except Exception as e:
    print('ERR', e)
