import os
import sys
import json

def truncateDataFile(filename):
    if(os.path.exists(filename)):
        ouputFile = open(filename, 'w')
        ouputFile.truncate()
        ouputFile.close()

def count_lines(filename):
    count = 0
    with open(filename, 'r') as file:
        for line in file:
            count += 1
    return count

input_path = "/mnt/vol/log-count-data.txt"
part_name = '%s/%s.txt'
output_folder = "/mnt/vol/split-output"

if not os.path.exists(output_folder):
        os.mkdir(output_folder)

numParts = int(os.environ['NUM_PARTS'])
totalLines = count_lines(input_path)
linesPreFile = totalLines / numParts

c = 0
num = 0

inputFile = open(input_path, 'r')
truncateDataFile(part_name % ( output_folder, num ))
outputFile = open(part_name % ( output_folder, num ), 'a')

while True:
    line = inputFile.readline()

    if not line:
        break

    if c < linesPreFile:
        outputFile.write('%s' % line)
    else:
        outputFile.close()
        c=0
        num+=1
        truncateDataFile(part_name % ( output_folder, num ))
        outputFile = open(part_name % ( output_folder, num ), 'a')
        outputFile.write('%s' % line)
    c+=1

inputFile.close()
outputFile.close()

partIds = list(map(lambda x: str(x), range(numParts)))
json.dump(partIds, sys.stdout)
