import cv2

replacement = ["replacement/1719900147189342620.jpg", "replacement/a432a0203df97130017ce13d1f512b84.jpg",
    "replacement/artworks-O14CPlEE4cs0wHGO-ohIOsQ-t500x500.jpeg", "replacement/download (1).jpeg", "replacement/download (2).jpeg", 
    "replacement/download (3).jpeg", "replacement/download (4).jpeg", "replacement/download (5).jpeg", "replacement/download (6).jpeg", 
    "replacement/download (7).jpeg", "replacement/download (8).jpeg", "replacement/download.jpeg", "replacement/images (1).jpeg",
    "replacement/images (2).jpeg", "replacement/images (3).jpeg", "replacement/images (4).jpeg", "replacement/images (5).jpeg",
    "replacement/images (6).jpeg", "replacement/images (7).jpeg", "replacement/images (8).jpeg", "replacement/images.jpeg"]

imgInp = cv2.imread("input/ugliest-people-in-the-world-8-62628152.jpg")
orb = cv2.ORB_create()
kp1, desInp = orb.detectAndCompute(imgInp, None)

for i in replacement:
    imgRep = cv2.imread(i, 0)
    kp2, desRep = orb.detectAndCompute(imgRep, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desRep, desInp)
    print ("Matches : ", len(matches), " ----- ", i)
    