import os

DIR_NAME = 'exams2'

names = set()
for filename in os.listdir(DIR_NAME):
    if '.jpeg' in filename:
        filename = filename.replace('-', ' ').replace('.jpeg', '')
        index = max([i for i in range(len(filename)) if filename[i].isalpha()])
        filename = filename[0:index + 1]
        names.add(filename)

f = open('roster-exams2', 'w')
for name in names:
    f.write('%s\n' % name)
f.close()