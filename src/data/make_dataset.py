# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
import pandas as pd
import re

# Disable the FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(input_dir, output_dir):
    """ Runs data loading scripts to turn SPSS datasets from (../raw) into
        CSV files (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making CSV files from SPSS datasets\n')

    input_files = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if f.endswith('.sav')]
    print(f'SPSS datasets to be converted: \n{input_files}\n') 

    for f in input_files:
        print(f'Converting `{f}`')
        print('...')
        df = pd.read_spss(f)

        year = re.search(r'\d{2,4}', f).group()[-2:]
        assert type(year) != list(), f'Variable `year` should not be a list, otherwise check the filename of {f}'
        
        output_name = f'{output_dir}/opd20{year}.csv'
        df.to_csv(f'{output_name}', index=False)
        print(f'`{output_name}` exported\n')


if __name__ == '__main__':
    import argparse

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(description='Load SPSS datasets and convert to CSV files')
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help='Directory to the input SPSS datasets'
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help='Directory to save the output CSV files'
    )
    args = parser.parse_args()    

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    main(input_dir, output_dir)