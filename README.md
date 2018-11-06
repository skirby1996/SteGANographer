# SteGANographer
Steganography is a cryptographic process where we attempt to hide a file or message inside
of another vessel file, usually an image or audio file, such that the vessel file appears
unchanged. A traditional steganographical algorithm, such as Least Significant Bit steganography
does this by encoding each bit for a file inside the least significant bit of the vessel file. 

Our project attempts to use neural networks to learn their own novel form of steganography. 

## TODO:
* Write a simple function to return an image capacity based on its dimensions
* Pad file batches with dummy bits so that each image batch has an associated message batch
* Pad image with whitespace so that trim is included in batches to encode. Trim should not
  count towards image capacity as padded whitespace will be removed from embedded image.
* Bring back Eve as an adversary attempting to recover bits, Alice and Bob's goal
  should be a 50% bit reconstruction rate while Eve hopes to achieve 100%


## Recommended IDE:
Visual Studio Code

## Recommended Extensions (Inside VSCode):
Click the Extensions panel
Search for "@recommended"
Install the Workspace Recommended extensions

## To install packages:
pip install -r requirements.txt
### OR (With VSCode):
Click Tasks -> Run Tasks -> Install requirements
