import argparse
import os
import ruamel_yaml as yaml
import json

import warnings
warnings.filterwarnings('ignore')

def main(args):
    total = 0
    number_of_yn_ques = 0
    yn_right = 0
    if (args.inference == 'rank'):
        output_data = None
        with open(f'{args.output_dir}/result/vqa_result.json') as fi:
            output_data = json.load(fi)
        for item in output_data:
            total += 1
            if (item['ground_truth'] in ['yes', 'no']):
                number_of_yn_ques += 1
                yn_right += int(item['ground_truth'] == item['model_answer'])
        with open(f'{args.output_dir}/result/evaluation_result.json', 'w') as fo:
            json.dump({'total' : total, 'yn_ques' : number_of_yn_ques, 'yn_right' : yn_right, 'yn_right_rate' : yn_right/number_of_yn_ques}, fo)
    else:
        output_data = None
        with open(f'{args.output_dir}/result/vqa_result.json') as fi:
            output_data = json.load(fi)
        for item in output_data:
            total += 1
            if (item['ground_truth'] in ['yes', 'no']):
                number_of_yn_ques += 1
                yn_right += int(item['ground_truth'] == item['model_answer'])
        with open(f'{args.output_dir}/result/evaluation_result.json', 'w') as fo:
            json.dump({'total' : total, 'yn_ques' : number_of_yn_ques, 'yn_right' : yn_right, 'yn_right_rate' : yn_right/number_of_yn_ques}, fo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference', default='rank')
    parser.add_argument('--output_dir', default='output')
    args = parser.parse_args()

    main(args)