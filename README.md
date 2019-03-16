# Crash Risk Calculator

Calculates the risk of an accident from 3 basic inputs.

Works in Chrome, Safari, Firefox, and soon in Edge (does not work, yet, as `tf.data` uses https://developer.mozilla.org/en-US/docs/Web/API/TextDecoder which is not support in Edge, https://github.com/tensorflow/tfjs/issues/1395)

App hosted here: https://djcordhose.github.io/crash-risk-calculator/

Source code: https://github.com/DJCordhose/crash-risk-calculator

Main source for training: https://github.com/DJCordhose/crash-risk-calculator/blob/master/train.js

Main source for prediction: https://github.com/DJCordhose/crash-risk-calculator/blob/master/calculator.js

## Data

The data set consists of 1500 samples for three categories, 500 samples each:

*  0 - red: many accidents
*  1 - green: few or no accidents
*  2 - yellow: in the middle

You have three variables to infer the category from
1. Top speed of car
1. Age of driver
1. Miles per year

The data: https://raw.githubusercontent.com/DJCordhose/deep-learning-crash-course-notebooks/master/data/insurance-customers-1500.csv

## Overview

You can use https://pair-code.github.io/facets/ to bucketize the data using speed and age

<img src='data/buckets.png'>

which already gives you a good overview how you would score each bucket.

