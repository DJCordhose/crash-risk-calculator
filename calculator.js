
const resultField = document.querySelector('#result');

const inSpeed = document.querySelector('#speed');
const inAge = document.querySelector('#age');
const inMiles = document.querySelector('#miles');

const calcBtn = document.querySelector('#calc-btn');

// init
let model = null;
(async () => {
  model = await tf.loadLayersModel('indexeddb://insurance');
  model.predict(tf.tensor([[48, 100, 10]])).print();
  calcBtn.disabled = null;
})();

function clearButtons() {
    const buttons = document.querySelectorAll('button.risk');
    buttons.forEach(btn => btn.style.display = 'none');
}

async function calculate(speed, age, miles) {
  const prediction = model.predict(tf.tensor([[age, speed, miles]]));
  const value = await prediction.data();
  return value;
}

calcBtn.addEventListener('click', async () => {
    const speed = parseFloat(inSpeed.value);
    const age = parseFloat(inAge.value);
    const miles = parseFloat(inMiles.value);

    const result = await calculate(speed, age, miles);

    // 0 - red: many accidents
    // 1 - green: few or no accidents
    // 2 - yellow: in the middle
    const [red, green, yellow] = result;
    console.log(result);

    clearButtons();
    if (red > green && red > yellow) {
        document.querySelector('button.btn-danger').style.display = 'inline';
    }
    if (yellow > green && yellow > red) {
        document.querySelector('button.btn-warning').style.display = 'inline';
    }
    if (green > red && green > yellow) {
        document.querySelector('button.btn-success').style.display = 'inline';
    }
});

