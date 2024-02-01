import os
import json

input_path = '/mnt/vol/count-output'
input_file = '%s/%s.json'

output_path = '/mnt/vol/result.json'

# 1. get numParts
numParts = int(os.environ['NUM_PARTS'])

# 2. parse, calculate
m = {
    'INFO': 0,
    'WARN': 0,
    'ERROR': 0,
    'DEBUG': 0
}

for i in range(0,numParts):
    with open(input_file % (input_path, i)) as f:
        obj = json.loads(f.read())

    for (k,v) in obj.items():
        m[k] += v

print('merge result:')
print(m)

with open(output_path, 'w') as f:
    f.write(json.dumps(m))
