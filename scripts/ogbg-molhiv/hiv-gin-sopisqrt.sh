#!/bin/bash
code=main_ogbg.py
python $code  --device ${1} --config "configs/ogbg-molhiv-ccp-sr1-128-32-gin-isqrt.json"
python $code  --device ${1} --config "configs/ogbg-molhiv-ccp-sr1-128-32-gin-isqrt.json"
python $code  --device ${1} --config "configs/ogbg-molhiv-ccp-sr1-128-32-gin-isqrt.json"
python $code  --device ${1} --config "configs/ogbg-molhiv-ccp-sr1-128-32-gin-isqrt.json"
python $code  --device ${1} --config "configs/ogbg-molhiv-ccp-sr1-128-32-gin-isqrt.json"


python main_ogbg.py  --device 0 --config="configs/ogbg-molhiv-ccp-sr1-128-64-gin-isqrt.json"
python main_ogbg.py  --device 0 --config="configs/ogbg-molpcba-CCP-SR1-128-64-32-gin-virtual-isqrt.json"