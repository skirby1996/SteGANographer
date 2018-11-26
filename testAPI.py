import sys
import os

import requests

import random
import string

import datetime

'''calls the /test GET endpoint of the API'''
def call_test_get():
    test_response = requests.get('https://stegosaurus.ml/api/test')
    print(test_response.status_code)

'''calls the /test POST endpoint of the API'''
def call_test_post():
    parameters = {'message': 'this is a test'}
    test_response = requests.post('https://stegosaurus.ml/api/test', params=parameters)
    print(test_response.status_code)
    print(test_response.content)

''' calls the /get_capacity endpoint of the API
    image: vessel file to check capacity of
    returns the capacity of image
'''
def call_get_capacity(image):

        files = {'image':(image, open(image, 'rb'))}
        parameters = {'formatted':'false'}

        capacity_response = requests.post('https://stegosaurus.ml/api/get_capacity', params=parameters, files=files)

        '''print(capacity_response.status_code)
        print("Content: " + capacity_response.content)'''

        return capacity_response.content


''' calls the /insert endpoint of the API
    image: vessel file that will contain encrypted content
    content: file that will be embedded into vessel image
    key:used for client side encryption and Neural Network
    returns a link to the image containing encrypted content
'''
def call_insert(image, content, key):

    files = {'image':(image, open(image, 'rb')), 'content':(content, open(content, 'rb'))}
    parameters = {'key':key}

    insert_response = requests.post('https://stegosaurus.ml/api/insert', params=parameters, files=files)

    '''print(insert_response.status_code)
    print("Content: " + insert_response.content)'''

    return insert_response.content

''' calls the /extract endpoint of the API
    img_url: url of the image containing encrypted content
    key: used for client side encryption and Neural Network
'''
def call_extract(img_url, key):

    parameters = {'image_url':img_url[1:-2], 'key': key}

    extract_response = requests.post('https://stegosaurus.ml/api/extract', params=parameters)

    '''print(extract_response.status_code)'''
    return extract_response.content

''' generates a file to embed into vessel image
    filename: name of file to write out to
    size: number of random characters to write to file
'''
def generate_file_to_embed(filename, size):

    chars = ''.join([random.choice(string.letters) for i in range(size)]) #1

    with open(filename, 'w') as f:
        f.write(chars)
    pass

def main():

    test_img_dir = "TestImages"
    content_file = 'content_file.txt'
    key = 'key'

    '''temporary --- path for local testing'''
    root_dir = os.path.abspath('')
    in_dir = os.path.join(root_dir, test_img_dir)

    '''create file to write results to'''
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    results_file = 'results_' + date_time + '.txt'


    listing = os.listdir(in_dir)
    for file in listing:
        img_dir = os.path.join(in_dir, file)

        '''generate file to embedded, fill it with random number of random characters'''
        content_size = random.randint(1000, 10000)
        generate_file_to_embed(content_file, content_size)

        img_capacity = call_get_capacity(img_dir)

        if(img_capacity > content_size):

            img_url = call_insert(img_dir, content_file, key)
            returned_content = call_extract(img_url, key)

            if returned_content == open(content_file).read():
                with open(results_file, 'a+') as f:
                    f.write("Content File Size: "+ str(content_size) +", " + file + ":\tSUCCESS \n")
                pass
            else:
                with open(results_file, 'a+') as f:
                    f.write("Content File Size: "+ str(content_size) +", " + file + ":\tFAIL \n")
                pass

        else:
            with open(results_file, 'a+') as f:
                f.write("Content File Size: "+ str(content_size) +", " + file + ":\tFAIL - content too large \n")
            pass

if __name__ == "__main__":
	main()
