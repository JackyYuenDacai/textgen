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

html = fetch('https://raw.githubusercontent.com/openai/openai-python/main/src/openai/types/responses/response_stream_event.py')
print(html[:3000])
