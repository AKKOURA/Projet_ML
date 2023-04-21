function afficherGraphe(graphId) {
  var maladies = [];
  var checkboxes = document.querySelectorAll('input[type=checkbox]:checked');

  for (var i = 0; i < checkboxes.length; i++) {
    maladies.push(checkboxes[i].value);
  }

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

function loadPage() {
  // Sélectionne toutes les cases à cocher et les coche
  var checkboxes = document.getElementsByName("class-select");
  for (var i = 0; i < checkboxes.length; i++) {
    checkboxes[i].checked = true;
  } 
  afficherGraphe('1'); 
  for (var i = 0; i < checkboxes.length; i++) {
    checkboxes[i].checked = false;
  } 

}

const graph1Btn = document.getElementById('graph1-btn');
graph1Btn.addEventListener('click', () => {
  afficherGraphe('1');
});

const graph2Btn = document.getElementById('graph2-btn');
graph2Btn.addEventListener('click', () => {
  afficherGraphe('2');
});

// const graph3Btn = document.getElementById('graph3-btn');
// graph3Btn.addEventListener('click', () => {
//   afficherGraphe('3');
// });

// const graph4Btn = document.getElementById('graph4-btn');
// graph4Btn.addEventListener('click', () => {
//   afficherGraphe('4');
// });

const graph5Btn = document.getElementById('graph5-btn');
graph5Btn.addEventListener('click', () => {
  afficherGraphe('5');
});

const graph6Btn = document.getElementById('graph6-btn');
graph6Btn.addEventListener('click', () => {
  afficherGraphe('6');
});

const graph7Btn = document.getElementById('graph7-btn');
graph7Btn.addEventListener('click', () => {
  afficherGraphe('7');
});