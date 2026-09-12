path = r'F:\GitHub\textgen\tests\test_responses_api.py'
src = open(path, encoding='utf-8').read()
old = "import asyncio\nimport copy\nimport json\nimport threading\nimport time\nimport unittest\n"
new = "import asyncio\nimport copy\nimport json\nimport threading\nimport unittest\n"
assert src.count(old) == 1
src = src.replace(old, new)
open(path, 'w', encoding='utf-8', newline='').write(src)
print('cleaned')
