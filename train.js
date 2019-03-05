// import * as tf from '@tensorflow/tfjs';
// import * as tfvis from '@tensorflow/tfjs-vis';

(async () => {
  const parser = d3.dsvFormat(';');

  // step 1: load and pre-process data
  const r = await fetch(
    'https://raw.githubusercontent.com/DJCordhose/deep-learning-crash-course-notebooks/master/data/insurance-customers-1500.csv'
  );
  const csv = await r.text();
  const data = await parser.parse(csv);

  const inputData = data.map((entry) => {
    return [
      Number(entry.age),
      Number(entry.speed),
      Number(entry.miles)
    ];
  });
  // does this look good?
  console.log(inputData[0]);

  const groups = data.map((entry) => {
    return Number(entry.group);
  });

  // 0 - red: many accidents
  // 1 - green: few or no accidents
  // 2 - yellow: in the middle
  const [red, green, yellow] = [0, 1, 2];

  // https://github.com/tensorflow/tfjs-vis
  // https://storage.googleapis.com/tfjs-vis/mnist/dist/index.html
  // https://github.com/tensorflow/tfjs-vis#renderscatterplotdata--container-surfacehtmlelement-opts---promise

  const valuesAgeSpeedRed = data.filter(entry => Number(entry.group) === red).map(entry => ({x: entry.age, y: entry.speed}));
  const valuesAgeSpeedGreen = data.filter(entry => Number(entry.group) === green).map(entry => ({x: entry.age, y: entry.speed}));
  const valuesAgeSpeedYellow = data.filter(entry => Number(entry.group) === yellow).map(entry => ({x: entry.age, y: entry.speed}));
  const scatterContainer = document.getElementById('scatter-surface');
  tfvis.render.scatterplot({
    values: [valuesAgeSpeedRed, valuesAgeSpeedGreen, valuesAgeSpeedYellow],
    series: ['many accidents', 'few or no accidents', 'in the middle']
  }, scatterContainer, {
xLabel: 'Age',
yLabel: 'Speed',
fontSize: 24,
zoomToFit: true,
height: 500
  });

  // step 2: define a simple sequential model
  const model = tf.sequential();
  model.add(tf.layers.dense({ name: 'hidden1', units: 100, activation: 'relu', inputShape: [3] }));
  model.add(tf.layers.dense({ name: 'hidden2', units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ name: 'softmax', units: 3, activation: 'softmax' }));
  model.summary();

   // https://github.com/tensorflow/tfjs-vis#showmodelsummarycontainer-drawable-model-tfmodel--promise
   const modelContainer = document.getElementById('model-surface');
   tfvis.show.modelSummary(modelContainer, model);

  model.compile({
    loss: 'categoricalCrossentropy',
    optimizer: 'adam',
    metrics: ['accuracy']
  });

  // step 3: train the model using our data
  const X = tf.tensor2d(inputData, [inputData.length, 3]);
  const y = tf.oneHot(tf.tensor1d(groups, 'int32'), 3);

  const surfaceContainer = document.getElementById('metrics-surface');
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const vizCallbacks = tfvis.show.fitCallbacks(surfaceContainer, metrics);
  const consoleCallbacks = {
    onEpochEnd: (...args) => {
      console.log(...args);
    },
    onEpochBegin: (iteration) => {
      // console.clear();
      console.log(iteration);
    }
  };

  const history = await model.fit(X, y, {
    epochs: 50, // was 500, down to 50 to make this train quickly
    validationSplit: 0.2,
    batchSize: 500, // was 500, down to 100 to try on mobile
    callbacks: vizCallbacks
    // callbacks: consoleCallbacks
  });

  // tfvis.show.history(surfaceContainer, history, ['loss', 'val_loss', 'acc', 'val_acc'],
  // {
  //   width: 1000,
  //   height: 300,
  // });

  // step 4: evaluate the model

  const {acc, loss, val_acc, val_loss} = history.history;
  const summary = `accuracy: ${acc[acc.length -1]}, accuracy on unknown data: ${val_acc[val_acc.length -1]}`;
  console.log(summary);
  const summaryContainer = document.getElementById('summary');
  summaryContainer.textContent = summary;
  model.predict(tf.tensor([[48, 100, 10]])).print();

  // 0 - red: many accidents
  // 1 - green: few or no accidents
  // 2 - yellow: in the middle
  const classNames = ['many accidents', 'few or no accidents', 'in the middle']; 
  const labels = tf.tensor(groups);
  const preds = model.predict(X).argMax([-1]);

  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const matrixContainer = document.getElementById('matrix-surface');
  tfvis.show.confusionMatrix(matrixContainer, confusionMatrix, classNames);

  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const accuracyContainer = document.getElementById('accuracy-surface');
  tfvis.show.perClassAccuracy(accuracyContainer, classAccuracy, classNames);

  // step 5: safe to local browser storage
  // https://js.tensorflow.org/tutorials/model-save-load.html

  // localstorage will not work with a large model (even 500/500 layer size will be too much in chrome)
  // model.save('localstorage://insurance');
  model.save('indexeddb://insurance');
  console.log(await tf.io.listModels());
})();
