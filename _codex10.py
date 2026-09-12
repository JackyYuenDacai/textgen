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

for path in ['codex-rs/protocol/src/models.rs', 'codex-rs/api/src/lib.rs']:
    try:
        src = fetch('https://raw.githubusercontent.com/openai/codex/main/' + path)
        print('=====', path, len(src))
        for m in re.finditer(r'FunctionCallOutput', src):
            i = m.start()
            print(src[max(0,i-600):i+700])
            print('~~~~~')
            break
    except Exception as e:
        print(path, 'ERR', e)
