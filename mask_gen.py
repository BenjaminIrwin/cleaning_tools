parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', type=str, default='/content/test', help='source')
parser.add_argument('--txt_output_dir', type=str, default='/content/output', help='txt_output_dir')
parser.add_argument('--mask_output_dir', type=str, default='/content/output', help='mask_output_dir')
parser.add_argument('--include_cutout', action='store_true', help='include_cutout')
parser.add_argument('--cutout_output_dir', type=str, default='/content/output', help='cutout_output_dir')
parser.add_argument('--delete_txt_files', action='store_true', help='delete_txt_files')

# Load all the text files
text_files = glob.glob(args.text_output_dir + '/*.txt')
print('Found {} text files'.format(len(text_files)))
for t in text_files:
    # Parse text file to get the bounding boxes
    with open(t, 'r') as f:
        lines = f.readlines()
        # convert each line to a list of floats
        mask_xyxy = [line.strip().strip('[]').split(', ') for line in lines]
        print(lines)
        size = img.size

        img = Image.new("RGB", size, (0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle(mask_xyxy, fill="white")
        # Delete the text file
        if args.delete_txt_files:
            os.remove(t)