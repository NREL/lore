#! /bin/bash

usage="A script to download and process historical sky-cover forecast data from NDFD.

Usage:
  download_ndfd.sh HAS_ID lat long

Download all files associated with NDFD request HAS_ID, extract the forecast
files into an intermediate form, constructs a file sky_cover.csv containing the
aggregated results, interpolated at the point lat-long.

Examples:
  download_ndfd.sh HAS011359513 38.23 -117.36

Obtaining a HAS_ID:

To obtain as HAS_ID, order historical NDFD data through the archive information
request system:

https://www.ncei.noaa.gov/has/HAS.FileAppRouter?datasetname=9959_02&subqueryby=STATION&applname=&outdest=FILE

The WMO headers are explained here:

  https://www.weather.gov/media/mdl/ndfd/NDFDelem_fullres_201906.xls

but we want YAQ:

  Y    (GRIB regional use)
  A    (Sky cover)
  Q    (Geographical area designator: Regional CONUS)

Choose sensible start and end dates (there is going to be a lot of data),
provide an email address, and click 'Proceed With Order'.

Upon order completion, you will be emailed a key like 'HAS011359513'. This is
your HAS_ID.

Sometimes there is too much data for NDFD to handle in a single request. If this
happens, you will need to create muptiple requests covering the time-window.

Run this script for each of the requests, then form a unified sky_cover.csv by
concatenating the csv files:
  cat HAS*/sky_cover.csv > sky_cover.csv"

# Check we have a copy of degrib
_SOURCE_DIR="$(dirname ${BASH_SOURCE[0]} > /dev/null 2>&1 ; pwd -P)"
if [[ "$OSTYPE" == "darwin"* ]]; then
    DEGRIB="$_SOURCE_DIR/degrib_osx"
else
    echo "ERROR: we don't have a version of degrib for the OS $OSTYPE"
    exit 1
fi
echo "INFO: found a copy of degrib at $DEGRIB"

# Step 0: check that the correct arguments have been provided, and make a new
#   directory for the HAS_ID.
if [ $# -ne 3 ]
then
  echo "$usage"
  exit 1
else
  HAS_ID="${1}"
  LATITUDE="${2}"
  LONGITUDE="${3}"
  echo "INFO: processing $HAS_ID at (latitude,longitude)=($LATITUDE,$LONGITUDE)"
fi

if [ ! -d $HAS_ID ]
then
    mkdir $HAS_ID
else
    echo "WARN: directory $HAS_ID already exists. Using that instead."
fi
cd $HAS_ID

# Step 1: download raw files from NOAA.
if [ ! -d "raw" ]
then
    echo "INFO: downloading files into $HAS_ID/raw..."
    mkdir "raw"
    cd "raw"
    wget --mirror --no-parent \
        --base=https://www1.ncdc.noaa.gov/pub/has/$HAS_ID/ \
        -F -i https://www1.ncdc.noaa.gov/pub/has/$HAS_ID/
    cd ".."
else
    echo "WARN: files in $HAS_ID/raw already downloaded. Using those instead."
fi

# Step 2: untar forecast files into intermediate directory.
if [ ! -d "intermediate" ]
then
    echo "INFO: extracting files into $HAS_ID/intermediate..."
    mkdir "intermediate"
    cd "intermediate"
    for filename in ../raw/www1.ncdc.noaa.gov/pub/has/$1/*.tar
    do
        tar xf $filename
    done
    cd ".."
else
    echo "WARN: files already extracted into $HAS_ID/intermediate. Using those instead."
fi

# Step 3: use degrib to interpolate the forecast at the provided latitude and
#   longitude.
if [ ! -d "processed" ]
then
    echo "INFO: processing files into into $HAS_ID/processed..."
    mkdir "processed"
    cd "processed"
    for filename in ../intermediate/*
    do
        $DEGRIB $filename -P -pnt $LATIDUDE,$LONGITUDE -out ${filename##*/}
    done
    echo "INFO: finished extracting!"
    cd ".."
else
    echo "WARN: files already processed into $HAS_ID/processed. Using those instead."
fi

# Step 4: aggregate the results into a single csv.
if [ ! -f "sky_cover.csv" ]
then
    echo "INFO: aggregating files..."
    cd processed
    find . -maxdepth 1 -type f -name '*.prb' -print0 | sort -zV | xargs -0 cat > out.tmp
    mv out.tmp ../out.tmp
    cd ..
    sed '/element/d' out.tmp > aggregated.txt
    sed 's/Sky, \[\%\], //g' aggregated.txt > sky_cover.csv
    rm out.tmp
    rm aggregated.txt
    echo "INFO: success! Created the file sky_cover.csv"
else
    echo "WARN: the file sky_cover.csv already exists. Use that instead."
fi
