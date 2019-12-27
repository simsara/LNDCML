import csv
import os
import pandas as pd

def readCSV(filename):
    lines = []
    # with open(filename, "rb") as f:
    with open(filename, "rt") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

annotations_filename = '/home/lwq/dataset/luna16/CSVFILES/seriesuids.csv'
annotations = readCSV(annotations_filename)
print('annotations_excluded length : ',len(annotations))
#print(annotations[4][0])

fold_path = '/home/lwq/dataset/luna16/subset/subset0'
all_files = os.listdir(fold_path)
fold_id = []
for f in all_files:
    if f.endswith('.mhd'):
        id = f[:-4]
        #print('single id : ', id)
        fold_id.append(id)
#print(fold_id)
#print(annotations[4])

fold_anno_file = '/home/lwq/dataset/luna16/CSVFILES/uid_subset/uid_subset0.csv'
fold_anno = []
# print(annotations[0])

for fi in fold_id:
    #print(fi)
    for i in range(len(annotations)):
        if fi == annotations[i][0]:
            #print(i)
            fold_anno.append(annotations[i])

print('subset annoex length : ',len(fold_anno))

fold_anno = pd.DataFrame(columns=annotations[0], data=fold_anno)

fold_anno.to_csv(fold_anno_file, encoding='utf-8')








