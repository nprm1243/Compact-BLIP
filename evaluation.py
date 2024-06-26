import argparse
import os
import numpy as np
import ruamel_yaml as yaml
import json

import warnings
warnings.filterwarnings('ignore')

def main(args):
    total = 0
    number_of_closed_ques = 0
    open_right = 0
    closed_right = 0
    output_data = None
    with open(f'{args.output_dir}/result/vqa_result.json') as fi:
        output_data = json.load(fi)
    for item in output_data:
        total += 1
        if (item['ground_truth'] in ['yes', 'no']):
            number_of_closed_ques += 1
            closed_right += int(item['ground_truth'] == item['model_answer'])
        else:
            open_right += int(item['ground_truth'] == item['model_answer'])
    with open(f'{args.output_dir}/result/evaluation_result.json', 'w') as fo:
        json.dump({'total' : total, 'closed_ques' : number_of_closed_ques, 'closed_right' : closed_right, 'open_right' : open_right, 'closed_accuracy' : np.round((closed_right/number_of_closed_ques)*100.00, 6), 'open_accuracy': np.round((open_right/(total - number_of_closed_ques))*100.0, 6), 'total_accuracy':np.round((open_right + closed_right)/total*100.0, 6)}, fo)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference', default='rank')
    parser.add_argument('--output_dir', default='output')
    args = parser.parse_args()

    main(args)