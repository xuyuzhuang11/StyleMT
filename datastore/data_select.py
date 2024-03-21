# import Levenshtein
import sentencepiece as spm
import sys

if __name__ == "__main__":
    path = sys.argv[1]
    lan_1 = sys.argv[2]
    lan_2 = sys.argv[3]
    lan_3 = sys.argv[4]

    spm_model = "mbart-spm.model"
    spm_proc = spm.SentencePieceProcessor(model_file=spm_model)

    for i in ["0", "1", "2", "3", "4", "5"]:
        fi1 = open(path + "/" + i + "/" + i + "." + lan_1, "r", encoding="utf-8")
        fi2 = open(path + "/" + i + "/" + i + "." + lan_2, "r", encoding="utf-8")
        fi3 = open(path + "/" + i + "/" + i + "." + lan_3, "r", encoding="utf-8")
        fo1 = open(path + "/" + i + "/" + i + "." + lan_1 + ".out", "w", encoding="utf-8")
        fo2 = open(path + "/" + i + "/" + i + "." + lan_2 + ".out", "w", encoding="utf-8")
        fo3 = open(path + "/" + i + "/" + i + "." + lan_3 + ".out", "w", encoding="utf-8")

        while True:
            l_zh1 = fi1.readline()
            if l_zh1 == "":
                break

            l_en = fi2.readline()
            l_zh2 = fi3.readline()
            l_zh1 = l_zh1.strip()
            l_en = l_en.strip()
            l_zh2 = l_zh2.strip()
            
            if max(len(l_zh1), len(l_zh2)) / min(len(l_zh1), len(l_zh2)) >= 5:
                continue

            fo1.write(l_zh1 + "\n")
            fo2.write(l_en + "\n")
            fo3.write(l_zh2 + "\n")

        fi1.close()
        fi2.close()
        fi3.close()
        fo1.close()
        fo2.close()
        fo3.close()
