function afficherGraphe(graphId) {
  fetch('/graph' + graphId)
    .then(response => response.json())
    .then(data => {
      Plotly.newPlot('graph-container', data.data, data.layout);
    });
}

const graph1Btn = document.getElementById('graph1-btn');
graph1Btn.addEventListener('click', () => {
  afficherGraphe('1');
});

const graph2Btn = document.getElementById('graph2-btn');
graph2Btn.addEventListener('click', () => {
  afficherGraphe('2');
});

const graph3Btn = document.getElementById('graph3-btn');
graph3Btn.addEventListener('click', () => {
  afficherGraphe('3');
});

const graph4Btn = document.getElementById('graph4-btn');
graph4Btn.addEventListener('click', () => {
  afficherGraphe('4');
});