import os, sys

if __name__ == "__main__":
    for fname in (os.listdir("./") ):
        if not fname.endswith(".txt"):
            continue
        name = fname[:-4]
        names = name.split("_")
        newname = names[2] + "_" + names[0][3:] + ".txt"
        print(newname)
        os.rename(fname, newname)
