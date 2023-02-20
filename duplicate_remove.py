import fastdup
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='data/images', help='path to images')
parser.add_argument('--work_dir', type=str, default='out', help='path to working directry')
parser.add_argument('--delete_duplicates', action='store_true', help='add argument to delete duplicates at the end')


def main():
    args = parser.parse_args()
    work_dir = args.work_dir
    input_dir = args.input_dir
    dry_run = not (args.delete_duplicates)

    if not dry_run:
        print('You are NOT doing a dry run, all of the duplicates will be deleted at the end')
        
    fastdup.run(input_dir=input_dir, work_dir=work_dir, nearest_neighbors_k=8,
                turi_param='ccthreshold=0.96')  # main running function.
    fastdup.create_components_gallery(work_dir, save_path='.')
    fastdup.create_duplicates_gallery(work_dir, save_path='.')
    top_components = fastdup.find_top_components(work_dir)
    
    if not dry_run:
        res = fastdup.delete_components(top_components, None, how='one', dry_run=dry_run)
        print('image deleted:')
        print(res)
    else:
        fastdup.delete_components(top_components, None, how='one', dry_run=dry_run)

if __name__ == '__main__':
    main()
