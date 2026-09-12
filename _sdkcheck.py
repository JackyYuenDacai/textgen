import urllib.request, re
url = 'https://raw.githubusercontent.com/openai/openai-python/main/src/openai/resources/responses/responses.py'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
r = urllib.request.urlopen(req, timeout=60)
html = b''
while True:
    chunk = r.read(65536)
    if not chunk:
        break
    html += chunk
html = html.decode('utf-8', 'ignore')
print('len', len(html))
events = sorted(set(re.findall(r"response\.[a-z_]+\.[a-z_]+", html)))
print('events:', events)
i = html.find('def _handle_event')
print(html[i:i+3000] if i != -1 else 'no _handle_event')
