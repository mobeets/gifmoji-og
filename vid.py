# src: https://www.makeartwithpython.com/blog/building-a-snapchat-lens-effect-in-python/

import time
import argparse
import os.path
import numpy as np
import cv2
import dlib
from imutils.video import VideoStream
from imutils import face_utils, translate, resize
from scipy.spatial import distance
from gifmoji import get_emojis_and_clrs, emoji_inds_to_img, get_image_block_clrs

"""
for more intensive frame processing...
https://github.com/npinto/opencv/blob/master/samples/python2/video_threaded.py
https://github.com/jrosebr1/imutils/blob/master/imutils/video/webcamvideostream.py
"""

def process_face((x,y,h,w), facelayer, emojis, eclrs):
    """
    could eventually find face angle, and try to 
    project emojis onto that plane...!
    see four-point transform in imutils, or FaceAligner

    note: facelayer will be [height x width], so x and y are swapped
    """
    # get facelayer of the correct size, e.g., (64x64x3)
    # make sure size is divisible by emoji size
    w = facelayer.shape[1]-x if x+w > facelayer.shape[1] else w
    h = facelayer.shape[0]-y if y+h > facelayer.shape[0] else h
    ew, eh = emojis[0].shape[1:]
    nw = ew*int(w/ew)
    nh = eh*int(h/eh)
    curface = facelayer[y:(y+nh),x:(x+nw),:]
    if len(curface) == 0 or curface.shape[0]*curface.shape[1] == 0:
        return facelayer

    # get avg color in each block of curface
    clrs, shape = get_image_block_clrs(curface, (ew,eh,3))

    # process by picking out best emoji for each tile
    dists = distance.cdist(clrs, eclrs, 'euclidean')
    inds = np.argmin(dists, axis=-1)
    indsb = inds.reshape(shape[0], shape[1])

    # group emojis into appropriate shape
    img = emoji_inds_to_img(emojis[indsb])
    
    # put newface back into facelayer in the appropriate spot
    facelayer[y:(y+nh),x:(x+nw),:] = img
    return facelayer

def get_frame(vs, args):
    frame = vs.read()
    width = frame.shape[1]
    height = frame.shape[0]
    frame = frame[args.trim_height:height-args.trim_height:, args.trim_width:width-args.trim_width,:]
    frame = resize(frame, width=args.width)
    return frame

def main(args):
    emojis, eclrs = get_emojis_and_clrs(args.emojifile, args.emoji_scale, args.n_to_trim)

    vs = VideoStream().start()
    time.sleep(1.5)

    # this detects our face
    detector = dlib.get_frontal_face_detector()

    recording = False
    processFace = True
    processAll = False
    counter = 0

    # get our first frame outside of loop, so we can see how our
    # webcame resized itself, and it's resolution w/ np.shape
    frame = get_frame(vs, args)

    facelayer = np.zeros(frame.shape, dtype='uint8')
    facemask = facelayer.copy()
    facemask = cv2.cvtColor(facemask, cv2.COLOR_BGR2GRAY)
    translated = np.zeros(frame.shape, dtype='uint8')
    translated_mask = facemask.copy()

    while True:
        # read a frame from webcam, resize to be smaller
        frame = get_frame(vs, args)

        # fill our masks and frames with 0 (black) on every draw loop
        facelayer.fill(0)
        facemask.fill(0)
        translated.fill(0)
        translated_mask.fill(0)

        # the detector and predictor expect a grayscale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        # if we're running the processFace loop
        # (press 's' while running to enable)
        if processFace and not processAll:
            for rect in rects:
                # draw bounding box around face
                x,y,w,h = face_utils.rect_to_bb(rect)
                # cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 128), 2)
                # frame = process_face((x,y,w,h), frame, emojis, eclrs)

                # get face mask
                faceBox = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
                cv2.fillPoly(facemask, [faceBox], 255)

                # copy image from frame onto facelayer using mask
                facelayer = cv2.bitwise_and(frame, frame, mask=facemask)
                facelayer = process_face((x,y,w,h), facelayer, emojis, eclrs)

            # again, cut out the translated mask
            frame = cv2.bitwise_and(frame, frame, mask=255-facemask)
            # and paste in the translated eye image
            frame += facelayer

        if processAll:
            frame = process_face((0,0,frame.shape[0],frame.shape[1]), frame, emojis, eclrs)

        # display current frame, and check if user pressed a key
        cv2.imshow("gifmoji", frame)
        key = cv2.waitKey(1) & 0xFF

        if recording:
            # create a directory called "image_seq" and
            # we'll be able to create gifs in ffmpeg from image sequences
            fnm = os.path.join(args.outdir, "%05d.png" % counter)
            cv2.imwrite(fnm, frame)
            counter += 1

        if key == ord("q"):
            break

        if key == ord("s"):
            processFace = not processFace

        if key == ord("a"):
            processAll = not processAll

        if key == ord("r"):
            recording = not recording

    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="movies",
        help="path to save pngs from movie")
    parser.add_argument("--width", type=int, default=1000,
        help="size of screen")
    parser.add_argument("--trim_width", type=int, default=200,
        help="size of screen on sides to trim")
    parser.add_argument("--trim_height", type=int, default=100,
        help="size of screen on top/bottom to trim")
    parser.add_argument("--emoji_scale", type=float, default=0.5,
        help="scale of 1.0 means (32x32) emojis")
    parser.add_argument('--n_to_trim', type=int, default=1,
        help="e.g., 1 -> emojis are 30x30 instead of 32x32")
    parser.add_argument('--emojifile', type=str,
        default='images/emojis.png')
    args = parser.parse_args()
    main(args)
