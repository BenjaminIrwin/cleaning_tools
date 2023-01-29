import boto3
import argparse

aws_access_key_id = 'AKIAW3YIJRQ7653XVP6F'
aws_secret_access_key = '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'

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, default='img', help='prefix of images')
parser.add_argument('--output_folder', type=str, default='data/images', help='path to images')


def main():
    args = parser.parse_args()
    output_folder = args.output_folder
    prefix = args.prefix

    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                             aws_secret_access_key=aws_secret_access_key)

    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket='lightsketch-bucket', Prefix=prefix)

    urls = []

    image_index = 0

    for response in pages:
        for content in response.get('Contents', []):
            img_name = 'a_' + str(image_index) + '.jpg'
            dest = output_folder + img_name
            print(dest)
            s3_client.download_file('lightsketch-bucket', content['Key'], dest)
            image_index += 1

    print(urls)


if __name__ == "__main__":
    main()
