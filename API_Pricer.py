import requests
import json
import os
import numpy as np

## Get Proxy ##
proxy = 'http://settanan.t:Tofu1412$@172.16.238.201:9090'

os.environ['http_proxy'] = proxy 
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy
proxies = {"http":"http://settanan.t:Tofu1412$@172.16.238.201:9090"}


## Get Token ##
# data = {
#         'username': 'KsecAPI',
#         'password': 'Welcome1!'
#         }

# url = 'https://api.idd.pt.ice.com/ipa/Api/v1/Authenticate'
# r = requests.get(url, json=data, proxies=proxies)

# token_key = r.json()['token']


## JSON Object ##
jsonString = '''
{
    "Valuation": {
      "Type": "Realtime"
    },
    "Artifacts": {
        "Instruments": [
            "Spread",
            "Instrument",
            "Theta"
        ],
        "UnderlyingAssets": {
            "EQ": [
                "Delta",
                "Vega",
                "Gamma"
            ]
        }
    },
    "Instruments": [
      {
        "InstrumentType": "Autocallable",
        "Id": 1,
        "BuySell": "Sell",
        "PayoutCurrency": "THB",
        "StrikeDate": "2023-12-01",
        "ExpiryDate": "6m",
        "IssueDate": "14d",
        "FirstCouponDate": "1m",
        "CouponFrequency": "Monthly",
        "BarrierType":"WorstOf",
        "UnderlyingAssets": [],
        "Notional": 1000000,
        "CouponStyle": "Binary",
        "CouponMemory": "No",
        "CouponPerAnnum":false,
        "AutocallBarrier": {
          "Continuous": false,
          "Direction": "Above",
          "Immediate": false,
          "Level": 100
        },
        "CouponBarrier": {
          "Continuous": false,
          "Direction": "Above",
          "Level": 0
        },
        "DayCountBasis":"30360",
        "OptionsAtExpiry": {
          "OptionAtExpiryBarrierType": "WorstOf",
          "Options":  [
              {
                "InstrumentType": "Vanilla",
                "CallPut": "Put",
                "Strike": 100,
                "BuySell": "Sell",
                "Gearing": 100
              }],
          "PerformanceType":"WorstOf"
        }
      }
    ]
}
'''
jsonObject = json.loads(jsonString)

## Get Date ##
StrikeDate = '2023-12-01'
ExpiryDate = '6m'
FirstCouponDate = '1m'
IssueDate = '14d'

jsonObject['Instruments'][0]['StrikeDate'] = StrikeDate
jsonObject['Instruments'][0]['ExpiryDate'] = ExpiryDate
jsonObject['Instruments'][0]['FirstCouponDate'] = FirstCouponDate
jsonObject['Instruments'][0]['IssueDate'] = IssueDate

## Get Notional ##
Notional = 1000000.
jsonObject['Instruments'][0]['Notional'] = Notional

## Get UL ##
InitialSpot = [25.25, 3.74]
Tickers = ['BGRIM', 'AWC']
bbTickers = [x + ' TB' for x in Tickers]
sdTickers = [x + '.BK' for x in Tickers]

for i in range(len(Tickers)):
    UL = {
        "SDTicker": sdTickers[i],
        "BBGTicker": bbTickers[i],
        "InitialSpot": InitialSpot[i],
        "Weight": 100
        }

    jsonObject['Instruments'][0]['UnderlyingAssets'].append(UL)

## Get KI ##
KI_barrier = 60.
jsonObject['Instruments'][0]['OptionsAtExpiry']['Options'][0]['Strike'] = KI_barrier

## Get Autocall Barrier ##
KO_barrier = 100.
stepdown = 1
tenor = 6
strike = 100.
Rate = 0.025

ind = np.arange(0, tenor, 1)
barrier = np.arange(KO_barrier, KO_barrier-(stepdown*tenor)+1, -1)
barrier = np.append(barrier, strike)

jsonObject['Instruments'][0]["AutocallBarrier"]['Level'] = KO_barrier


# for i in range(len(barrier)):
#     coupons = {"AutocallBarrier":barrier[i],
#                "AutoRedemption":100.0,
#                "CouponBarrier":0.0,
#                "Date": ,
#                "GuaranteedCoupon":0.0,
#                "Rate":Rate}

#     jsonObject['Instruments'][0]['Coupons'].append(coupons)
    
## Call API ##
url = 'https://api.idd.pt.ice.com/EQ/Api/v1/Calculate'
headers = {'Content-Type': 'application/json', 
           'AuthenticationToken': '4b9f9134-6c20-49c2-aebf-9fb663eb7d1b'}
r = requests.post(url, headers=headers, json=jsonObject)

## Get Result ##
Result = json.loads(r.text)
# print(Result['instruments'][0]['price'])
print(Result['instruments'][0]['instrument']['fixings'])