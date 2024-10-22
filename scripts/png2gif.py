import os, sys
import imageio

png_dir = sys.argv[1]
start_frame=int(sys.argv[2])
end_frame=int(sys.argv[3])

images = []
#for file_name in sorted(os.listdir(png_dir)):
for i in range(start_frame, end_frame):
    file_name = str(i) + ".png"
    file_path = os.path.join(png_dir, file_name)
    images.append(imageio.imread(file_path))

# Make it pause at the end so that the viewers can ponder
for _ in range(10):
    images.append(imageio.imread(file_path))

imageio.mimsave('../movie.gif', images, fps=60)
