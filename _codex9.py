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

src = fetch('https://raw.githubusercontent.com/openai/codex/main/codex-rs/core/src/client_common.rs')
open(r'F:\GitHub\textgen\_codex_common.rs', 'w', encoding='utf-8').write(src)
print('len', len(src))
for pat in ['FunctionCallOutput', 'function_call_output', 'previous_response_id', 'ResponseItem::', 'store', 'instructions']:
    print(pat, len(re.findall(pat, src)))
