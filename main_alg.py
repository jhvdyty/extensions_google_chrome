import imagehash
from PIL import Image

replacement = ["replacement/1719900147189342620.jpg", "replacement/a432a0203df97130017ce13d1f512b84.jpg",
    "replacement/artworks-O14CPlEE4cs0wHGO-ohIOsQ-t500x500.jpeg", "replacement/download (1).jpeg", "replacement/download (2).jpeg", 
    "replacement/download (3).jpeg", "replacement/download (4).jpeg", "replacement/download (5).jpeg", "replacement/download (6).jpeg", 
    "replacement/download (7).jpeg", "replacement/download (8).jpeg", "replacement/download.jpeg", "replacement/images (1).jpeg",
    "replacement/images (2).jpeg", "replacement/images (3).jpeg", "replacement/images (4).jpeg", "replacement/images (5).jpeg",
    "replacement/images (6).jpeg", "replacement/images (7).jpeg", "replacement/images (8).jpeg", "replacement/images.jpeg"]

imgInp = Image.open("input/ugliest-people-in-the-world-8-62628152.jpg")

hashInp = imagehash.phash(imgInp)


for i in replacement:
    imgRep = Image.open(i)
    hashRep = imagehash.phash(imgRep)
    print ("Difference: ", hashRep - hashInp, " ----- ", i)