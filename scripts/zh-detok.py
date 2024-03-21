import sentencepiece as spm
import sys

if __name__ == "__main__":
    spm_model = "mbart-spm.model"
    spm_proc = spm.SentencePieceProcessor(model_file=spm_model)

    for l in sys.stdin:
        print(spm_proc.decode(l.strip().split()))
        

