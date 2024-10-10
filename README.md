# TimeParticle

TimeParticle: Particle-like Multiscale State Space Models for Time Series Forecasting

## Get Started

1. Install requirements. ```pip install -r requirements.txt```
2. Download data. You can download the all datasets from [datasets](https://drive.google.com/u/0/uc?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download). **All the datasets are well pre-processed** and can be used easily.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:

```bash
bash ./scripts/long_term_forecast/ETT_script/TimeParticle_ETTm1.sh
bash ./scripts/long_term_forecast/ECL_script/TimeParticle.sh
bash ./scripts/long_term_forecast/Traffic_script/TimeParticle.sh
bash ./scripts/long_term_forecast/Weather_script/TimeParticle.sh
```



