

def write_log(msg, file_num):
    fo = open("ant_log_%s.txt" % file_num, "a")
    fo.write(msg)
    fo.write("\n")
    fo.close()
