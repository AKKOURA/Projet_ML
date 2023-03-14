import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-hello-world',
  templateUrl: './hello-word.component.html',
})
export class HelloWorldComponent implements OnInit {
  message: string | undefined;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.http.get('http://127.0.0.1:5000/hello').subscribe((data: any) => {
      this.message = data.message;
      console.log(data)
    });
  }
}
