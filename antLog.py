def write_log(msg):
    fo = open("ant_lot.txt", "a")
    fo.write(msg)
    fo.write("\n")
    fo.close()
