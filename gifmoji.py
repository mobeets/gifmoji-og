import os.path
import argparse
import numpy as np
from skimage.util.shape import view_as_blocks
from scipy.misc import imread
from PIL import Image
from scipy.spatial import distance
EMOJI_SIZE = 32

def array_to_Image(img):
    return Image.fromarray(np.swapaxes(img, 0, 1))

def Image_to_array(img):
    return np.swapaxes(np.array(img), 0, 1)

def Image_rgba_to_rgb(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.
    src: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)
    """
    image.load() # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background

def load_emojis(infile='images/emojis.png', upsample=1.0):
    img = imread(infile, mode='RGBA')
    B = view_as_blocks(img, block_shape=(EMOJI_SIZE, EMOJI_SIZE, 4))
    B = B[:,:,0,:,:,:]
    C = np.reshape(B, (B.shape[0]*B.shape[1], EMOJI_SIZE, EMOJI_SIZE, 4))
    return [Image.fromarray(c).resize((int(upsample*c.shape[0]), int(upsample*c.shape[1]))) for c in C]

def load_target(infile='images/trump.png', upsample=1.0):
    img = Image.open(infile).convert('RGB')
    img = img.resize((upsample*img.size[0], upsample*img.size[1]))
    return Image_to_array(img)

def fitness(yh, y):
    return -np.sqrt(np.square(1.0*y - 1.0*yh).sum())

def encode(img):
    return 1.0*np.array(Image_rgba_to_rgb(img)).flatten()

def get_positions((lx, ly), (sx, sy), no_grid, grid_stride, num_iters):
    if no_grid:
        # sample locations randomly
        rng = np.random.RandomState(666)
        pos = (rng.rand(num_iters, 2)*Y.shape[:2]).astype(int)
    else:
        # make grid of locations, with some overlap
        xs = np.arange(0, lx, sx/grid_stride)
        ys = np.arange(0, ly, sy/grid_stride)
        px, py = np.meshgrid(xs, ys)
        pos = np.vstack([px.flatten(), py.flatten()]).T
    # randomly re-order, and add jitter
    inds = np.random.permutation(pos.shape[0])
    pos = pos[inds]
    pos += args.sigma*np.random.randn(*pos.shape)
    return pos.astype(int)

def get_img_block(Y, (cx,cy), (sx,sy)):
    Yc = Y[cx:cx+sx,cy:cy+sy]
    nx = sx - Yc.shape[0]
    ny = sy - Yc.shape[1]
    if nx > 0 or ny > 0:
        # pad incomplete image blocks with mean value
        Yc = np.pad(Y[cx:cx+sx,cy:cy+sy],
            [(0,nx), (0,ny), (0,0)], mode='mean')
    cur_img_block = Yc.flatten()
    return 1.0*cur_img_block

def rescore(Y, Yc):
    # Yhc = np.pad(np.array(Yh)[cx:cx+sx,cy:cy+sy],
    #   [(0,nx), (0,ny), (0,0)],
    #   mode='mean')#, constant_values=255)
    # Yhc = Image.fromarray(Yhc)
    for j in range(len(emojis)):
        # Yhcc = Yhc.copy()
        # Yhcc.paste(emojis[j], (0,0), emojis[j])
        # em = Image_rgba_to_rgb(Yhcc)
        # if use_pixel_loss:
        #   resps[j] = np.array(em).flatten()
        # else:
        #   resps[j] = imgfcn(np.array(em))
        scores[j] = fitness(resps[j], target_block)

# TRUMP_INDS = [17, 21, 24, 42, 47, 54, 61, 65, 67, 73, 84, 86, 90, 93, 101, 106, 107, 110, 114, 115, 118, 134, 138, 140, 143, 144, 148, 155, 165, 169, 170, 173, 177, 195, 197, 204, 205, 225, 233, 234, 243, 263, 287, 293, 294, 323, 334, 354, 375, 405, 465, 472, 476, 536, 556, 592, 593, 616, 623, 653, 683, 684, 706, 712, 713, 736, 740, 743, 764, 763, 767, 802, 803, 804, 830, 834, 863, 864, 881, 893]
# TRUMP_INDS = [891, 893, 863, 834, 802, 804, 773, 764, 743, 740, 736, 713, 712, 706, 684, 683, 653, 616, 593, 563, 556, 536, 465, 439, 405, 375, 354, 334, 323, 293, 294, 291, 264, 263, 234, 233, 225, 203, 204, 195, 173, 148, 118, 114, 107, 106, 90, 86, 73, 54, 47, 42, 33, 21]
# emojis = [e for i,e in enumerate(emojis) if i in TRUMP_INDS]

def emoji_encode(img):
    Yc = 1.0*np.array(Image_rgba_to_rgb(img))
    return np.reshape(Yc, (EMOJI_SIZE*EMOJI_SIZE, -1)).mean(axis=0)

def main_fast(args):
    """
    if we just want a perfect grid of emojis, we can use this

    here, we just compare based on average color of each block
    """
    # load target image, extract segments of image
    Y = load_target(args.targetfile, upsample=args.target_upsample)
    wx = EMOJI_SIZE*(Y.shape[0]/EMOJI_SIZE)
    wy = EMOJI_SIZE*(Y.shape[1]/EMOJI_SIZE)
    Y = Y[:wx,:wy,:].transpose(1,0,2)
    Ys = view_as_blocks(Y, block_shape=(EMOJI_SIZE,EMOJI_SIZE,3))

    # find mean color of each block
    pts = Ys[:,:,0,:,:,:]
    pts = pts.transpose(0,1,-1,2,3)
    pts = np.reshape(pts, pts.shape[:-2] + (-1,))
    clrs = np.mean(pts, axis=-1) # w x h x 3
    clrs = np.reshape(clrs.transpose(-1,0,1), (3, -1)).T # w*h x 3

    # load emojis, and find mean color of each emoji
    emojis = load_emojis(args.emojifile, upsample=args.unit_upsample)
    emojis = [1.0*np.array(Image_rgba_to_rgb(img)) for img in emojis]
    emojis = np.array(emojis)
    emojis = emojis.transpose(0,-1,1,2)
    eclrs = np.reshape(emojis, emojis.shape[:2] + (-1,))
    eclrs = np.mean(eclrs, axis=-1)

    # find closest emoji in terms of color
    dists = distance.cdist(clrs, eclrs, 'euclidean')
    inds = np.argmin(dists, axis=-1)
    indsb = inds.reshape(pts.shape[0], pts.shape[1])

    # combine emojis to create final image
    E = emojis[indsb]
    E = E.transpose(0,1,3,4,2)
    Es = [E[:,:,:,:,i] for i in range(3)]
    Es2 = [B.transpose(0,2,1,3).reshape(-1,B.shape[1]*B.shape[3]) for B in Es]
    E2 = np.dstack(Es2).astype(np.uint8)
    img = Image.fromarray(E2, 'RGB')

    # save image
    outfile = os.path.join(args.outdir, args.run_name + '.png')
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    img.save(outfile)

def main(args):
    emojis = load_emojis(args.emojifile, upsample=args.unit_upsample)

    sx, sy = emojis[0].size
    if not args.silent:
        print('Loaded {} emojis.'.format(len(emojis)))

    Y = load_target(args.targetfile, upsample=args.target_upsample)
    # Image.fromarray(np.swapaxes(Y, 0, 1)).save(outfile.replace('.', '_target.'))
    if not args.silent:
        print('Loaded target image with shape {}.'.format(Y.shape))

    # find layer response to each 32x32 emoji
    emoji_blocks = [encode(img) for img in emojis]
    
    outfile = os.path.join(args.outdir, args.run_name + '.png')
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    # iterate through locations and find best emoji in that spot
    Yh = Image.new('RGBA', (Y.shape[:2]))
    pos = get_positions(Y.shape[:2], (sx, sy), args.no_grid,
        args.grid_stride, args.num_iters)
    if not args.silent:
        print('Trying {} positions.'.format(len(pos)))
    for i in range(len(pos)):
        # pick next location
        cx,cy = pos[i,:]
        target_block = get_img_block(Y, (cx,cy), (sx,sy))        

        # get current score
        if not args.force_add:
            cur_block = get_img_block(Image_to_array(Image_rgba_to_rgb(Yh)), (cx,cy), (sx,sy))
            null_score = fitness(cur_block, target_block)

        # find best emoji for current 32x32 target block
        scores = [fitness(blk, target_block) for blk in emoji_blocks]
        j = np.argmax(scores)

        # paste if this improves score, or if we have to do it anyway
        if args.force_add or scores[j] > null_score:
            Yh.paste(emojis[j], (cx,cy), emojis[j])
            if not args.silent and i % args.log_every == 0:
                print("Iter #{}, Emoji #{}, Score: {:0.2f}".format(i, j, scores[j]))
        else:
            if not args.silent and i % args.log_every == 0:
                print("Iter #{}, Emoji #{}, Score: {:0.2f}".format(i, np.nan, null_score))
        if i % args.save_every == 0:
            Yh.save(outfile)
    Yh.save(outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str,
        help='tag for current run')
    parser.add_argument('--targetfile', type=str,
        default='images/trump.png')
    parser.add_argument('--emojifile', type=str,
        default='images/emojis.png')
    parser.add_argument('--force_add', action='store_true')
    parser.add_argument('--no_grid', action='store_true')
    parser.add_argument('--do_fast', action='store_true')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--gif_partial_mode', action='store_true',
        help="set if gif is in partial mode")
    parser.add_argument('--is_gif', action='store_true')
    parser.add_argument('--num_iters', type=int, default=2000, 
        help='if --no_grid, the number of position samples')
    parser.add_argument('--grid_stride', type=float, default=1.3)
    parser.add_argument('--sigma', type=float, default=1.0,
        help='standard deviation of jitter to positions')
    parser.add_argument('--target_upsample', type=int, default=2)
    parser.add_argument('--unit_upsample', type=int, default=1)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--outdir', type=str, default='.')
    args = parser.parse_args()

    if args.is_gif:
        from gifextract import processImage
        print("Extracting pngs from gif...")
        outfiles = processImage(args.targetfile,
            is_partial_mode=args.gif_partial_mode)
        run_name = args.run_name
        for i, outfile in enumerate(outfiles):
            print("Processing {} of {} pngs...".format(i+1, len(outfiles)))
            args.targetfile = outfile
            args.run_name = run_name + '-{:02d}'.format(i)
            print(args.targetfile, args.run_name)
            if args.do_fast:
                main_fast(args)
            else:
                main(args)
        # now, combine .pngs into a single .gif:
        # convert -dispose previous -delay 1 *.png out.gif
    else:
        if args.do_fast:
            main_fast(args)
        else:
            main(args)
