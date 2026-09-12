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

# list codex repo core/src dir via GitHub API
api = fetch('https://api.github.com/repos/openai/codex/contents/core/src')
names = re.findall(r'"name": "([a-z_0-9]+\.rs)"', api)
print(names)
