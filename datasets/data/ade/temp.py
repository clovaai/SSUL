fname_target = "train_cls.txt"
fname = "train_cls_old.txt"

new_cls_list = []

with open(fname_target, "w") as f_new:
    with open(fname, "r") as f:
        txt_list = f.read().splitlines()

        for line in txt_list:
            line = line.split(" ")

            img = line[0]
            clss = list(map(int, line[1:]))
            if 0 in clss:
                clss.remove(0)
            clss = [cls-1 for cls in clss]

            new_line = img
            for cls in clss:
                new_line += " %d" % cls

            f_new.write(new_line + "\n")

