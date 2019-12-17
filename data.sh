#!/bin/bash

kaggle datasets download -d sakell/foodclassdata \

echo '{"username":"sakell","key":"121bd0932d4fab1172b2220d6cf787a6"}' > /root/.kaggle/kaggle.json \

kaggle datasets download -d sakell/foodclassdata \

unzip -qq foodclassdata.zip \

rm foodclassdata.zip
