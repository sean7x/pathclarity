# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
import pandas as pd

# Disable the FutureWarnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(input_dir, output_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    input_files = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if f.endswith('.sav')]
    print(input_files) 


if __name__ == '__main__':
    import argparse

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser(desription='Load SPSS datasets and convert to CSV files')
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