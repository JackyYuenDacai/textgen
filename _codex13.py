import re
src = open(r'F:\GitHub\textgen\_codex_client.rs', encoding='utf-8').read()
for pat in ['timeout', 'Timeout', 'read_timeout', 'connect_timeout']:
    for m in re.finditer(pat, src):
        i = m.start()
        line = src[max(0, i-200):i+200].replace('\n', ' ')
        print(pat, '::', line[:380])
        print('---')
