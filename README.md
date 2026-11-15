# Weather Forecaster

A neural network that forecasts global weather, scored on the WeatherBench benchmark.

## How It Works

The model learns from ERA5, a reconstruction of the global atmosphere every 6 hours since 1959. This project uses it on a 64 x 32 grid, where each cell is 5.625 degrees wide. Given the state of the atmosphere now, the network predicts the state 3 or 5 days later. It trains on 1979 to 2015, uses 2016 to decide when to stop training, and is scored on 2017 and 2018, which it never sees.

Scores are root mean squared error on three fields WeatherBench tracks: geopotential at 500 hPa (Z500, the mid-level pressure pattern that steers weather systems), temperature at 850 hPa (T850, about 1.5 km up) and temperature 2 m above the ground (T2M). Grid cells near the poles cover less area than cells at the equator, so every error is weighted by the cosine of its latitude.

A forecast score only means something next to baselines that use no model at all. Persistence says the weather stays as it is now. Climatology says it will be the long-term average. Weekly climatology uses the average for that week of the year.

The first network is the CNN from the WeatherBench paper: five 5x5 convolutions, 64 channels wide, 313,858 parameters. Longitude wraps around the globe, so each convolution pads the east edge of the map with the west edge instead of zeros.

## Results

Baselines on the 2017 to 2018 test years, next to the numbers published in the WeatherBench paper (Rasp et al. 2020, Table 2). Lower is better.

| Field | Baseline | This repo | Published | Difference |
|---|---|---|---|---|
| Z500 (m^2/s^2) | persistence 3d | 906 | 936 | -3.2% |
| Z500 (m^2/s^2) | persistence 5d | 1003 | 1033 | -2.9% |
| Z500 (m^2/s^2) | climatology | 1058 | 1075 | -1.6% |
| Z500 (m^2/s^2) | weekly climatology | 795 | 816 | -2.6% |
| T850 (K) | persistence 3d | 3.92 | 4.23 | -7.2% |
| T850 (K) | persistence 5d | 4.20 | 4.56 | -7.9% |
| T850 (K) | climatology | 5.35 | 5.51 | -2.9% |
| T850 (K) | weekly climatology | 3.26 | 3.50 | -7.0% |
| T2M (K) | persistence 3d | 2.60 | 3.00 | -13.2% |
| T2M (K) | persistence 5d | 2.88 | 3.27 | -12.0% |
| T2M (K) | climatology | 5.89 | 6.07 | -3.0% |
| T2M (K) | weekly climatology | 2.91 | 3.19 | -8.8% |

The baselines here come out lower than the published ones, by 2 to 3 percent for Z500 and more for temperature. The likely reason is the data version. This repo uses the WeatherBench 2 copy of ERA5, which averages each coarse cell from the fine grid. The original benchmark interpolated instead, which keeps more small-scale detail and so gives larger errors. The gap is biggest for T2M, the field with the most small-scale detail, which fits that explanation. It is not confirmed yet, see Limitations.

The first network on the same test years, next to the CNN in the WeatherBench paper and weekly climatology, the strongest baseline above. Each lead time is its own trained network.

| Field | Lead time | This repo | Published CNN | Weekly climatology |
|---|---|---|---|---|
| Z500 (m^2/s^2) | 3 days | 623 | 626 | 795 |
| Z500 (m^2/s^2) | 5 days | 740 | 757 | 795 |
| T850 (K) | 3 days | 2.67 | 2.87 | 3.26 |
| T850 (K) | 5 days | 3.12 | 3.37 | 3.26 |

The network beats every no-model baseline at both lead times. Against weekly climatology its Z500 error is 22 percent lower at 3 days and 7 percent lower at 5 days. The lead shrinks because the atmosphere gets harder to predict the further ahead you look, while climatology is equally good at any lead time.

It also lands within a few percent of the published CNN. That is a match, not an improvement. The baselines on this data version already score 2 to 8 percent lower than the published ones, and the gaps here are no larger.

Each network trained for 30 epochs on a free Colab T4 GPU, about 37 minutes per network.

## Limitations

- Scores are not exactly comparable to the published WeatherBench table until the data-version explanation above is checked against the original benchmark files.
- The grid is coarse. One cell is about 600 km wide at the equator, so the model cannot see storms, fronts or anything local.
- The first network only sees Z500 and T850 and predicts one fixed lead time per trained model.
- Both networks ran to the 30-epoch limit with the validation loss still falling slowly, so a longer run might score a little better.
- Training on a laptop GPU is slow, about 2 minutes per epoch for the small CNN.

## Files

- `src/weather_forecaster/era5.py` downloads ERA5 channels from the public WeatherBench 2 bucket and reads them back as memory-mapped arrays
- `src/weather_forecaster/metrics.py` latitude-weighted RMSE and anomaly correlation
- `src/weather_forecaster/baselines.py` persistence and climatology forecasts
- `src/weather_forecaster/dataset.py` stacks channels and normalizes them for training
- `src/weather_forecaster/models.py` the convolutional network
- `src/weather_forecaster/train.py` training loop and scoring
- `scripts/` command-line entry points for downloading, baselines and training
- `notebooks/colab_train.ipynb` runs the download and training on a free Google Colab GPU
- `tests/` unit tests for the downloader, metrics, baselines, normalization, model and training loop
- `results/` scores written by the scripts

## Tech Stack

- Python, PyTorch
- NumPy, Zarr, gcsfs
- pytest, ruff

## How to Run

```
python3 -m venv venv
source venv/bin/activate
pip install -e . pytest

python scripts/download.py --preset baselines   # about 1.5 GB into data/
python scripts/baselines.py
python scripts/train.py --lead-days 3
pytest
```

Training saves its progress after every epoch, so running the same command again continues an interrupted run from the last completed epoch.

To train on a free GPU instead of your own machine, open `notebooks/colab_train.ipynb` in Google Colab and choose Runtime > Run all.
