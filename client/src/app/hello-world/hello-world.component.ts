import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import Chart from 'chart.js/auto';


@Component({
  selector: 'app-hello-world',
  templateUrl: './hello-word.component.html',
})
export class HelloWorldComponent implements OnInit {
  message: string | undefined;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    // this.http.get('http://127.0.0.1:5000/hello').subscribe((data: any) => {
    //   this.message = data.message;
    //   console.log(data)
    // });

    const canvas = document.getElementById('myCanvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;

    const img = new Image();
    img.src = './assets/test_graph.png';

    img.onload = function() {
      const pattern = ctx.createPattern(img, 'repeat');
      const chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: ['1', '2', '3', '4', '5'],
          datasets: [{
            label: 'My Dataset',
            backgroundColor: pattern!,
            borderColor: 'rgb(255, 99, 132)',
            data: [2, 4, 6, 8, 10]
          }]
        },
        options: {}
      });
    };

  }
}
