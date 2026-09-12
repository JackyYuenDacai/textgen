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
i = src.find('pub enum ResponseInputItem')
j = src.find('}', src.find('ToolSearchOutput', i))
print(src[i-200:j+200])
