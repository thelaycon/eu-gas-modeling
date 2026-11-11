# EU gas storage model.

*Author: Balogun.*

Pulls the EU-aggregate daily storage series from GIE AGSI+ (and ALSI for LNG), back to 2016; demand-weighted HDD across six countries from Open-Meteo; the monthly IMF "Global price of Natural Gas, EU (TTF)" from FRED plus the latest Yahoo TTF front-month price. 

Compares the current storage year (from 1 July) with the previous five, regresses daily net withdrawals on HDD with monthly intercepts (calibrated on winters 2022/23 onward), reruns the rest of the winter under the weather of each past winter (cold=10th / normal=median / mild=90th percentile of projected 1 April storage), backtests 1 April projections leave-one-winter-out, sizes LNG needs vs a 1 April target, and correlates TTF with the storage deviation since 2023.

Usage: `python3 eu_gas_storage_model.py [--target 30] [--backtest-date 02-01]  [--cargo-twh 1.2] [--offline]`
