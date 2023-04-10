function afficherGraphe(graphId) {
  fetch('/graph' + graphId)
    .then(response => response.blob())
    .then(blob => {
      const img = document.createElement('img');
      img.src = URL.createObjectURL(blob);
      const graphContainer = document.getElementById('graph-container');
      graphContainer.innerHTML = ''; // Supprimer les anciennes images
      graphContainer.appendChild(img);
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
