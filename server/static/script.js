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

function updateProgress() {
  progressData.forEach(function (data) {
    var progressBar = document.getElementById("progress_band-like_infiltrate");
    var progressValue = document.getElementById("progress_value_band-like_infiltrate");

    progressBar.value = data.progress;
    progressValue.textContent = data.progress;
  });
}

// Appelez la fonction updateProgress() pour afficher les barres de progression initiales
updateProgress();


var progressData = [
  {
    name: "band-like_infiltrate",
    progress: 0, // La valeur de progression initiale (0, 1, 2 ou 3)
  },
  // Ajoutez d'autres objets de données de progression si nécessaire
];
var progressBar = document.getElementById("progress_band-like_infiltrate");

progressBar.addEventListener("input", function () {
  var value = parseInt(progressBar.value);

  // Vérifiez si la valeur est dans la plage autorisée (0, 1, 2 ou 3)
  if (value >= 0 && value <= 3) {
    progressData[0].progress = value;
    updateProgress();
  } else {
    progressBar.value = progressData[0].progress; // Rétablir la valeur précédente si la nouvelle valeur est invalide
  }
});

