# -*- coding: utf-8 -*-
import argparse
import sys
from typing import List
from transformers import MarianMTModel, MarianTokenizer

def get_label_doc(input_list):
    output_labels = []
    output_words = []
    for line in input_list:
        if len(line.strip().split('\t')) == 2:
            label, doc = line.rstrip().split('\t')
            output_labels.append(label)
            output_words.append(doc)
        else:
            print('Source file should contain two columns only')
    return output_labels, output_words

def chunk_list(src_list, n):
    # looping till length l 
    for i in range(0, len(src_list), n):
        yield src_list[i:i + n]

def flatten_list(src_list):
    flattened_list = [item for sublist in src_list for item in sublist]
    return flattened_list



def translate(src_lines, model=None, tokenizer=None):
    chunks = list(chunk_list(src_lines, 6))
    translations = []
    for i, chunk in enumerate(chunks):
        translated = model.generate(**tokenizer.prepare_translation_batch(chunk))
        tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        translations.append(tgt_text)
    return flatten_list(translations)

def write_to_file(output_file, src_labels, translated_words):
    for label, line in zip(src_labels, translated_words):
        output_file.write(f"{label}\t{line}\n")



def main():
    parser = argparse.ArgumentParser(description='translating using MarianMT in transformers huggingface library')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(),
                        help='The input file (defaults to stdin)')
    parser.add_argument('-o', '--output', default=sys.stdout.fileno(),
                        help='The output annotated file (defaults to stdout)')
    parser.add_argument('--encoding', default='utf-8',
                        help='The character encoding for input/output '
                             '(it defaults to UTF-8)')
    parser.add_argument('--src', help='The source language')
    parser.add_argument('--tgt', help='The target language')
    args = parser.parse_args()

    f = open(args.input, encoding=args.encoding)
    output_file = open(args.output, mode='w', encoding=args.encoding)

    model_name = f"Helsinki-NLP/opus-mt-{args.src}-{args.tgt}"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    with open(args.input, 'r', encoding=args.encoding) as f:
        src_lines = f.readlines()
        labels, words = get_label_doc(src_lines)
        tgt_text = translate(words, model, tokenizer)
        write_to_file(output_file, labels, tgt_text)


if __name__ == '__main__':
    main()
