import urllib.request, re
def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    r = urllib.request.urlopen(req, timeout=120)
    data = b''
    while True:
        chunk = r.read(65536)
        if not chunk:
            break
        data += chunk
    return data.decode('utf-8', 'ignore')

src = fetch('https://raw.githubusercontent.com/openai/codex/main/codex-rs/protocol/src/models.rs')
i = src.find('FunctionCallOutputPayload')
while i != -1:
    seg = src[max(0,i-200):i+900]
    if 'enum' in seg or 'struct' in seg:
        print(seg)
        print('~~~~~~~~')
    i = src.find('FunctionCallOutputPayload', i+1)
