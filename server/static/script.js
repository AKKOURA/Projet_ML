function afficherGraphe(graphId) {
  var maladies = [];
  var checkboxes = document.querySelectorAll('input[type=checkbox]:checked');

  for (var i = 0; i < checkboxes.length; i++) {
    maladies.push(checkboxes[i].value);
  }

  console.log(JSON.stringify({maladies : maladies}))

  fetch('/graph' + graphId, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({maladies : maladies})
  })
  .then(function(response) {
    return response.json();
  })
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

const graph5Btn = document.getElementById('graph5-btn');
graph5Btn.addEventListener('click', () => {
  afficherGraphe('5');
});