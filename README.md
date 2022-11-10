# DRC
## Usage
1. Install DRC

        git clone https://github.com/tim25651/DRC
        cd DRC/
        pip install .


1. Provide your data results in a CSV file (Mark values to exclude with an * as suffix)

2. Execute DRC

        drc -d DoseColumnIndex DataFile -o OutputDirectory

3. Plot is stored in OutputDirectory/DataFileBase_plot.png
4. Parameters are stored in OutputDirectory/DataFileBase_params.csv

## Test
1. Create test data

        drc-test TestDataFile

2. Execute DRC

        drc TestDataFile

## Acknowledgments
Thanks to yannabraham (https://gist.github.com/yannabraham/5f210fed773785d8b638)