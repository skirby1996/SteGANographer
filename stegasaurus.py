import sys
import os

import numpy as np
import tensorflow as tf

from PIL import Image
from src.config import Config

import cv2
import numpy as np
import hashlib
import math
import os


def nn_to_bin(batch):
    return np.around(batch).astype(int)


def generate_header(file_path):
    """
    Creates header data (size, hash, and name) for a specific file.
    4 bytes for size is more or less arbitrary. If we ever need to embed anything >4GB it can be changed.
    16 bytes is the constant length of an MD5 digest.
    Filenames are padded 255 bytes because that is the max length for a filename in a lot of file systems.
    Altogether it adds up to 275 bytes, which is padded to 384 so it can occupy 3 reserved batches/blocks.
    """
    # 4 bytes for filesize
    filesize = os.path.getsize(file_path)
    if filesize >= 2**32:
        raise Exception(
            "File too big. Header currently only works for files up to 4GB.")
    header_bytes = filesize.to_bytes(4, byteorder="big")

    # 16 bytes for hash
    with open(file_path, "rb") as file:
        header_bytes += hashlib.md5(file.read()).digest()

    # 255 bytes for filename
    header_bytes += os.path.basename(file_path).rjust(255).encode()

    # pad header to be exactly 3 batches in size
    header_bytes = header_bytes + b'\0' * (384 - len(header_bytes))
    return header_bytes


def parse_header(header_bytes):
    """
    Extracts a file's size, hash, and name from a byte string of header data (expecting at least 275 bytes).
    If anything is added or changed in the generate_header function that must be reflected here.
    """
    size = header_bytes[:4]
    size = int.from_bytes(size, byteorder="big")

    f_hash = header_bytes[4:20]

    f_name = header_bytes[20:275]
    try:
        f_name = f_name.decode()
    except:
        print(f_name)
        raise Exception(
            "Content could not be retrieved from image. (File name could not be parsed.)")

    return size, f_hash, f_name.strip()


def bytes_to_batch(byte_str):
    """
    Converts exactly 128 bytes to a 32x32x1 numpy array to be used as a batch for the neural network.
    Bytes are expanded into bits and converted to {-1.0, 1.0} in place of {0, 1}.
    """
    if len(byte_str) != 128:
        raise Exception(
            "bytes_to_batch takes exactly 128 bytes (got %d)" % len(byte_str))

    # convert to list of 1024 bits
    def bit_list(b): return [int(i) for i in list(bin(b)[2:].zfill(8))]

    def flatten(l): return [item for sublist in l for item in sublist]
    bits = flatten([bit_list(b) for b in list(byte_str)])

    # convert from binary bits to -1.0 and 1.0 for neural network
    bits = [-1.0 if b == 0 else 1.0 for b in bits]

    # fill numpy array
    batch = np.zeros((32, 32, 1))
    ptr = 0
    for x in range(32):
        for y in range(32):
            batch[x][y][0] = bits[ptr]
            ptr += 1
    return batch


def batch_to_bytes(batch):
    """
    The inverse of the bytes_to_batch function.  A 32x32x1 numpy array with values {-1.0, 1.0} is converted
    into a string of 128 bytes.
    """
    if batch.shape != (32, 32, 1):
        raise Exception("Got batch with invalid shape: %s" % str(batch.shape))

    # convert 1.0 and -1.0 values back to integer 1 and 0 bits
    def tobits(b): return 1 if b > 0.0 else 0
    batch = np.vectorize(tobits)(batch)

    # extract bits from batch and turn into bytes
    bits = ""
    byte_vals = []
    for x in range(32):
        for y in range(32):
            bits = "%s%d" % (bits, batch[x][y][0])
            if len(bits) >= 8:
                byte_vals.append(int(bits, 2))
                bits = ""
    return bytes(byte_vals)


def file_to_batches(file_path):
    """
    Reads file from given path and converts it into nx32x32x1 batches for the neural network
    First three batches hold the header data (file name, file size, and hash).
    """
    header_bytes = generate_header(file_path)

    filesize = os.path.getsize(file_path)
    # 128 = number of bytes in 32x32 bitplane, 3 additional batches for header
    n = math.ceil(filesize / 128) + 3
    batches = np.zeros((n, 32, 32, 1))

    with open(file_path, "rb") as file:
        # convert header to batches
        for i in range(3):
            batches[i] = bytes_to_batch(header_bytes[128*i:128*(i+1)])

        # convert image to batches
        for batch_no in range(3, n):
            buf = file.read(128)

            if len(buf) < 128:
                # pad with null bytes
                buf = buf + b'\0' * (128 - len(buf))

            batches[batch_no] = bytes_to_batch(buf)
    return batches


def batches_to_file(batches, output_dir):
    """
    Converts content batches recieved from the neural network into a file.
    File name is derived from the header data and the file is saved to the specified directory.
    """
    # get params from header (first 3 batches)
    header_bytes = b''
    for n in range(3):
        header_bytes += batch_to_bytes(batches[n])
    file_size, header_hash, file_name = parse_header(header_bytes)

    if file_size > (batches.shape[0] - 3) * 128:
        raise Exception(
            "Content could not be retrieved from image. File size in header is too large.")

    # get file data from the rest of the batches
    file_bytes = b""
    for n in range(3, batches.shape[0]):
        file_bytes += batch_to_bytes(batches[n])
        if len(file_bytes) >= file_size:
            break
    file_bytes = file_bytes[:file_size]

    # verify hash
    file_hash = hashlib.md5(file_bytes).digest()
    if file_hash != header_hash:
        print("Warning: Hashes do not match")
        # raise Exception(
        #    "Content could not be retrieved from image. Hashes do not match.")

    # save file
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, "wb") as file_out:
        file_out.write(file_bytes)


def image_to_batches(img):
    """
    Converts image into 32x32x3 batches for neural network.
    Removes trim on right and bottom (blocks smaller than 32x32).
    """
    h, w, _ = img.shape
    n_h = math.floor(h / 32)
    n_w = math.floor(w / 32)
    batches = np.zeros((n_h * n_w, 32, 32, 3))
    n = 0
    # decompose image into 32x32 blocks
    for x in range(n_w):
        for y in range(n_h):
            batches[n] = img[y*32:(y+1)*32, x*32:(x+1)*32]
            n += 1
    # convert pixels from byte values into floats
    batches = np.vectorize(lambda b: b / 255)(batches)
    return batches


def batches_to_image(batches, img):
    """
    Converts batches back into image blocks.
    Pastes 32x32 blocks onto original image (to preserve trim).
    """
    n = batches.shape[0]
    h, w, _ = img.shape
    n_h = math.floor(h / 32)
    n_w = math.floor(w / 32)
    img_out = img.copy()

    if n_h * n_w < n:
        raise Exception("Too many batches for the provided image.")

    # convert pixel floats back to byte values
    batches = np.vectorize(lambda f: round(f * 255))(batches)

    # paste batches onto image
    batch_no = 0
    for x in range(n_w):
        for y in range(n_h):
            img_out[y*32:(y+1)*32, x*32:(x+1)*32] = batches[batch_no]
            batch_no += 1
            if batch_no >= n:
                return img_out

###############################################################
#                                                             #
#   These will be the functions that are called from the API  #
#   Right now they're still missing some code (marked in the  #
#   functions with comments) but everything else works        #
#                                                             #
###############################################################


def insert(cfg, sess, img_path, file_path):
    """
    Inserts the file at file_path into the image at img_path using the neural network.
    Returns the image as an opencv numpy array.  Can be saved with: cv2.imwrite(path, image)
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_batches = image_to_batches(img)
    file_batches = file_to_batches(file_path)

    print(tf.shape(img_batches))
    print(tf.shape(file_batches))

    batches_out = sess.run('alice_out:0', feed_dict={
        'img_in:0': img_batches, 'msg_in:0': file_batches})

    img_out = batches_to_image(batches_out, img)
    return img_out


def extract(cfg, sess, img_path, output_dir):
    """
    Extracts a content file from the image at img_path using the neural network.
    On success the file is saved to output_dir (file name comes from header).
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_batches = image_to_batches(img)

    batches_out = sess.run('bob_vars_1/bob_eval_out:0',
                           feed_dict={'img_in:0': img_batches})

    batches_to_file(batches_out, output_dir)


def main():
    cfg = Config()

    # Prepare directories
    root_dir = os.path.abspath("")
    model_dir = os.path.join(root_dir, "models")
    model_dir = os.path.join(model_dir, cfg.MODEL_NAME)
    if not os.path.exists(model_dir):
        print("Error: No model exists at %s" % model_dir)
        return False

    target_dir = os.path.join(root_dir, "production")
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    in_dir = os.path.join(target_dir, "input")
    out_dir = os.path.join(target_dir, "output")
    if not os.path.exists(in_dir):
        os.mkdir(in_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Load model
    save_dir = os.path.join(model_dir, "data")
    log_dir = os.path.join(model_dir, "logs")

    # Use log file to infer current epoch
    current_epoch = 0
    log_file_name = os.path.join(log_dir, cfg.MODEL_NAME + "_log.csv")
    if os.path.isfile(log_file_name):
        with open(log_file_name, 'r') as log_file:
            lines = log_file.readlines()
            if len(lines) > 1:
                current_epoch = int(lines[-1].split(',')[0])
                print("Current epoch: %d\n" % current_epoch)
            else:
                print("Error: Cannot infer epoch from log file...")
                return False
    else:
        print("Error: No log file from which to infer epoch")
        return False

    # Restore saved weights
    weight_file_name = cfg.MODEL_NAME + '_train-' + str(current_epoch)
    meta_file_name = weight_file_name + '.meta'

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(
            os.path.join(save_dir, meta_file_name))
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))

        repeat = True
        while repeat:
            mode = None
            if mode != 'e' and mode != 'd':
                mode = input("Encrypt or Decrypt? (E/D): ").lower()
            if mode == 'e':
                # Embed using Alice net
                user_in = input("Please enter name of vessel image: ")
                vessel_img_path = os.path.join(in_dir, user_in)

                user_in = input("Please enter file to embed: ")
                target_file_path = os.path.join(in_dir, user_in)

                if os.path.exists(vessel_img_path) and os.path.exists(target_file_path):
                    img = insert(cfg, sess, vessel_img_path, target_file_path)
                    cv2.imwrite(vessel_img_path[:-4] + '_embedded.bmp', img)
                else:
                    if not os.path.exists(vessel_img_path):
                        print("Error: No image at %s" % vessel_img_path)
                    if not os.path.exists(target_file_path):
                        print("Error: No file at %s" % target_file_path)
            else:
                # Extract using Bob net
                user_in = input("Please enter name of embedded image: ")
                embedded_img_path = os.path.join(in_dir, user_in)
                if os.path.exists(embedded_img_path):
                    extract(cfg, sess, embedded_img_path, out_dir)
                else:
                    print("Error: No file at %s" % embedded_img_path)

            user_in = None
            if user_in != 'y' and user_in != 'n':
                user_in = input("Repeat? (Y/N): ").lower()
            if user_in == 'n':
                repeat = False


if __name__ == '__main__':
    main()
