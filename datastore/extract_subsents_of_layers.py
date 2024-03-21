# usage: python extract_subsents_of_layers.py ./train.luxun.all.multi.zh 3 4 5 7 8 10

import sys
import os
import sentencepiece as spm

if __name__ == "__main__":
    print("Extracting monolingual sub-sentences of 6 transformer layers...")
    file_path = sys.argv[1]
    up_bd1 = int(sys.argv[2])
    up_bd2 = int(sys.argv[3])
    up_bd3 = int(sys.argv[4])
    up_bd4 = int(sys.argv[5])
    up_bd5 = int(sys.argv[6])
    up_bd6 = int(sys.argv[7])
    folder_path = os.path.dirname(file_path)
    file_name_list = os.path.basename(file_path).split(".")
    if file_name_list[2] == "all":
        folder_path = folder_path + "/all-" + str(up_bd6)
        os.mkdir(folder_path)
    else:
        folder_path = folder_path + "/" + file_name_list[0]
        os.mkdir(folder_path)
    for i in range(0, 6):
        os.mkdir(folder_path + "/" + str(i))
    spm_model = "mbart-spm.model"
    spm_proc = spm.SentencePieceProcessor(model_file=spm_model)
    lan = file_name_list[4] + "1"
    # lan = file_name_list[2] + "1"                   # languege extensive name

    fi = open(file_path, "r", encoding="utf-8")
    fo1 = open(folder_path + "/0/0." + lan, "w", encoding="utf-8")
    fo2 = open(folder_path + "/1/1." + lan, "w", encoding="utf-8")
    fo3 = open(folder_path + "/2/2." + lan, "w", encoding="utf-8")
    fo4 = open(folder_path + "/3/3." + lan, "w", encoding="utf-8")
    fo5 = open(folder_path + "/4/4." + lan, "w", encoding="utf-8")
    fo6 = open(folder_path + "/5/5." + lan, "w", encoding="utf-8")
    sent_set1 = set()
    sent_set2 = set()
    sent_set3 = set()
    sent_set4 = set()
    sent_set5 = set()
    sent_set6 = set()

    count = 0
    while True:
        line = fi.readline()
        if line == "":
            break

        if count % 500 == 0:
            print(f"{count} subsents")
        count += 1

        maps = line.split("|||")
        for code_sent in maps:
            sent = code_sent.split("||")[1]
            sent = sent.strip().split(" ")
            sent = [s.strip() for s in sent]
            # for sentencepiece
            sent = " ".join(sent)
            sent = spm_proc.encode(sent.strip(), out_type=str)

            '''if "▁" in sent:
                sent.remove("▁")'''
            length = len(sent)
            sent = " ".join(sent)
            if length == 0 or length > up_bd6:
                continue
            elif 1 <= length <= up_bd1:
                if sent not in sent_set1:
                    sent_set1.add(sent)
                    fo1.write(sent + "\n")
            elif up_bd1 < length <= up_bd2:
                if sent not in sent_set2:
                    sent_set2.add(sent)
                    fo2.write(sent + "\n")
            elif up_bd2 < length <= up_bd3:
                if sent not in sent_set3:
                    sent_set3.add(sent)
                    fo3.write(sent + "\n")
            elif up_bd3 < length <= up_bd4:
                if sent not in sent_set4:
                    sent_set4.add(sent)
                    fo4.write(sent + "\n")
            elif up_bd4 < length <= up_bd5:
                if sent not in sent_set5:
                    sent_set5.add(sent)
                    fo5.write(sent + "\n")
            else:
                if sent not in sent_set6:
                    sent_set6.add(sent)
                    fo6.write(sent + "\n")

    fi.close()
    fo1.close()
    fo2.close()
    fo3.close()
    fo4.close()
    fo5.close()
    fo6.close()
    print("Extracting monolingual sub-sentences of 6 transformer layers OK!")
