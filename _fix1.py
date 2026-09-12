p = r'F:\GitHub\textgen\_roundtrip.py'
src = open(p, encoding='utf-8').read()
old = "final = json.loads(events[-1]['data'])"
new = "final = json.loads(events[-1]['data'])\nfinal = final.get('response', final)"
assert old in src
src = src.replace(old, new)
open(p, 'w', encoding='utf-8').write(src)
print('patched')
