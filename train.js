// import * as tf from '@tensorflow/tfjs';
// import * as tfvis from '@tensorflow/tfjs-vis';

const DATA_URL =
  "https://raw.githubusercontent.com/DJCordhose/deep-learning-crash-course-notebooks/master/data/insurance-customers-1500.csv";

// define model globally, so you can use it from the console later
let model;

(async () => {
  // step 1: load, plot and pre-process data

  // https://js.tensorflow.org/api/latest/#data.csv
  const data = await tf.data.csv(DATA_URL, {
    delimiter: ";",
    columnConfigs: {
      group: { isLabel: true }
    }
  }).shuffle(1500).toArray();

  // https://js.tensorflow.org/api/latest/#class:data.CSVDataset
  // const columnNames = await csvDataset.columnNames();
  // console.log(columnNames);

  // https://js.tensorflow.org/api/latest/#class:data.Dataset
  // 0 - red: many accidents
  // 1 - green: few or no accidents
  // 2 - yellow: in the middle
  const [red, green, yellow] = [0, 1, 2];

  // https://github.com/tensorflow/tfjs-vis
  // https://js.tensorflow.org/api_vis/latest/#render.scatterplot

  const valuesAgeSpeedRed = data
    .filter(({ xs, ys }) => Number(ys.group) === red)
    .map(({ xs, ys }) => ({ x: xs.age, y: xs.speed }));
  const valuesAgeSpeedGreen = data
    .filter(({ xs, ys }) => Number(ys.group) === green)
    .map(({ xs, ys }) => ({ x: xs.age, y: xs.speed }));
  const valuesAgeSpeedYellow = data
    .filter(({ xs, ys }) => Number(ys.group) === yellow)
    .map(({ xs, ys }) => ({ x: xs.age, y: xs.speed }));

  const scatterContainer = document.getElementById("scatter-surface");
  tfvis.render.scatterplot(
    {
      values: [valuesAgeSpeedRed, valuesAgeSpeedGreen, valuesAgeSpeedYellow],
      series: ["many accidents", "few or no accidents", "in the middle"]
    },
    scatterContainer,
    {
      xLabel: "Age",
      yLabel: "Speed",
      fontSize: 18,
      zoomToFit: true,
      height: 400,
      yAxisDomain: [80, 180],
      xAxisDomain: [15, 110]
    }
  );

  // step 2: define a simple sequential model
  // https://js.tensorflow.org/api/latest/#layers.dropout
  // https://js.tensorflow.org/api/latest/#layers.batchNormalization

  const DROP_OUT = 0.6;
  const REGULARIZE = true;
  model = tf.sequential();
  model.add(
    tf.layers.dense({
      name: "hidden1",
      units: 250,
      inputShape: [3]
    })
  );

  // https://towardsdatascience.com/checklist-for-debugging-neural-networks-d8b2a9434f21
  // https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
  // https://arxiv.org/abs/1801.05134
  model.add(tf.layers.activation({ activation: "relu" }));
  if (REGULARIZE) {
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.dropout({ rate: DROP_OUT }));
  }

  model.add(tf.layers.dense({ name: "hidden2", units: 250 }));
  model.add(tf.layers.activation({ activation: "relu" }));
  if (REGULARIZE) {
    model.add(tf.layers.batchNormalization());
    model.add(tf.layers.dropout({ rate: DROP_OUT }));
  }

  model.add(
    tf.layers.dense({ name: "softmax", units: 3, activation: "softmax" })
  );
  model.summary();

  // https://js.tensorflow.org/api_vis/latest/#show.modelSummary
  const modelContainer = document.getElementById("model-surface");
  tfvis.show.modelSummary(modelContainer, model);

  model.compile({
    loss: "sparseCategoricalCrossentropy",
    optimizer: "adam",
    metrics: ["accuracy"]
  });

  // step 3: train the model using our data
  const xs = data
    .map(({ xs, _ }) => Object.values(xs));

  const ys = data
    .map(({ _, ys }) => Object.values(ys)[0]);

  console.log('Number of datasets for fitting:', xs.length);  

  const X = tf.tensor2d(xs, [xs.length, 3]);
  // const y = tf.oneHot(tf.tensor1d(ys, 'int32'), 3);
  // one hot not needed when using sparse categorical crossentropy as loss
  const y = tf.tensor1d(ys, "int32");

  const surfaceContainer = document.getElementById("metrics-surface");
  const metrics = ["loss", "val_loss", "acc", "val_acc"];
  const vizCallbacks = tfvis.show.fitCallbacks(surfaceContainer, metrics);
  const consoleCallbacks = {
    onEpochEnd: (...args) => {
      console.log(...args);
    },
    onEpochBegin: iteration => {
      // console.clear();
      console.log(iteration);
    }
  };

  const BATCH_SIZE = 200;
  const EPOCHS = 50;

  // https://js.tensorflow.org/api/latest/#tf.LayersModel.fitDataset
  // https://js.tensorflow.org/api/latest/#tf.LayersModel.fit
  const history = await model.fit(X, y, {
    epochs: EPOCHS,
    validationSplit: 0.2,
    batchSize: BATCH_SIZE,
    shuffle: true,
    callbacks: vizCallbacks
    // callbacks: consoleCallbacks
  });

  // tfvis.show.history(surfaceContainer, history, ['loss', 'val_loss', 'acc', 'val_acc'],
  // {
  //   width: 1000,
  //   height: 300,
  // });

  // step 4: evaluate the model

  const { acc, loss, val_acc, val_loss } = history.history;
  const summary = `accuracy: ${
    acc[acc.length - 1]
  }, accuracy on unknown data: ${val_acc[val_acc.length - 1]}`;
  console.log(summary);
  const summaryContainer = document.getElementById("summary");
  summaryContainer.textContent = summary;
  model.predict(tf.tensor([[100, 48, 10]])).print();

  // 0 - red: many accidents
  // 1 - green: few or no accidents
  // 2 - yellow: in the middle
  const classNames = ["many accidents", "few or no accidents", "in the middle"];
  const yTrue = tf.tensor(ys);
  const yPred = model.predict(X).argMax([-1]);

  const confusionMatrix = await tfvis.metrics.confusionMatrix(yTrue, yPred);
  const matrixContainer = document.getElementById("matrix-surface");
  tfvis.show.confusionMatrix(matrixContainer, confusionMatrix, classNames);

  const classAccuracy = await tfvis.metrics.perClassAccuracy(yTrue, yPred);
  const accuracyContainer = document.getElementById("accuracy-surface");
  tfvis.show.perClassAccuracy(accuracyContainer, classAccuracy, classNames);

  // https://js.tensorflow.org/api_vis/latest/#show.layer
  tfvis.show.layer(
    document.getElementById("hidden1-layer-surface"),
    model.getLayer("hidden1")
  );
  tfvis.show.layer(
    document.getElementById("hidden2-layer-surface"),
    model.getLayer("hidden2")
  );
  tfvis.show.layer(
    document.getElementById("softmax-layer-surface"),
    model.getLayer("softmax")
  );

  // step 5: safe to local browser storage
  // https://js.tensorflow.org/tutorials/model-save-load.html

  // localstorage will not work with a large model (even 500/500 layer size will be too much in chrome)
  // model.save('localstorage://insurance');
  model.save("indexeddb://insurance");
  console.log(await tf.io.listModels());
})();
