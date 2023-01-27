import glob

text_files = glob.glob('boundings/*.txt')
print('Found {} text files'.format(len(text_files)))
for t in text_files:
    # Parse text file to get the bounding boxes
    with open(t, 'r') as f:
        lines = f.readlines()
        # convert each line to a list of floats
        lines = [list(map(float,line.strip().strip('[]').split(', '))) for line in lines]
        print(lines)