import re
src = open(r'F:\GitHub\textgen\_codex_client.rs', encoding='utf-8').read()
i = src.find('previous_response_id')
while i != -1 and i < len(src):
    print('--- at', i)
    print(src[max(0,i-800):i+400])
    i = src.find('previous_response_id', i+1)
    if i > 40000: break
