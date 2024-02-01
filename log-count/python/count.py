import os
import re
import json

# part_id: get form environment variable, should start from 0
part_id = os.environ['PART_ID']

input_path = '/mnt/vol/split-output/%s.txt' % (part_id)
output_path = '/mnt/vol/count-output'
output_file = '%s/%s.json' % (output_path, part_id)

if not os.path.exists(output_path):
    os.mkdir(output_path)

with open(input_path) as f:
    txt = f.read()

m = {
    'INFO': 0,
    'WARN': 0,
    'ERROR': 0,
    'DEBUG': 0
}
# count
for k in m:
    m[k] = len(re.findall(k, txt))

print(m)

with open(output_file, 'w') as f:
    f.write(json.dumps(m))
