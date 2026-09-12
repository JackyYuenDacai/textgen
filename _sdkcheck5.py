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
names = sorted(set(re.findall(r'(Response[A-Za-z]*Event)', html)))
print(names)
i = html.find('summary')
print('...context around summary...')
print(html[max(0,i-400):i+400])
