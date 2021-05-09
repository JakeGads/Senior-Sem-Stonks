# Senior-Sem-Stonks
A senior sem group project for DeSales University using a python neural network to predict stocks

## Running

Ensure [docker](https://www.docker.com/) is installed on your system.

if you are running windows double click the `docker_run.bat` 

if you are running MacOs, Linux, freeBSD or any derivative of Unix run `docker_run.sh`

NOTE: the initial setup will take longer because the docker image must download

When the app is running it will generate a url that will host the app [localhost:5000](http://localhost:5000)

NOTE: becuase we use some of the more advanced tensor flow features your computer may throw an warning, this is as expected and will not hamper results but does affect execution time

## API

When connected to the URL the following has the following paths, that all point to the same base page

```
/
/home
/home/<tags>
/home/<tags>/<startdate>
/home/<tags>/<startdate>/<enddate>
```

### TAGS

A tag is just the stock tag we can compile mutiple by seperating them with the `|`(pipe), we chose to the pipe because it was an unused character in html and hence considered safe example tags are

```
GME
GME|ACM
GME|ACM|A|AAC
```

If a tag doesn't exist an error message will be returned instead 

### STARTDATE
A Datetime str denoting the earliest date that is pulled from yahoo, there are many ways to format date strings in python but the defaulted way is:
```
YYYY-MM-DD
```
The defualted value is `2018-01-01`

### ENDDATE
A Datetime str denoting the latest date that is pulled from yahoo, there are many ways to format date strings in python but the defaulted way is:

```
YYYY-MM-DD
```
The defualted value is set to the current datetime of the server

The Enddate is not exposed on the front end


## CLI

We recommned that you add the project to your PATH from there it would be a simple

```
stonks-bot <tags> <startdate>
```

### TAGS

A tag is just the stock tag we can compile mutiple by seperating them with the ` `(space)

```
GME
GME ACM
GME ACM A AAC
```

If a tag doesn't exist an error message will be returned instead

### STARTDATE

A Datetime str denoting the earliest date that is pulled from yahoo, there are many ways to format date strings in python but the defaulted way is:
```
YYYY-MM-DD
```
The defualted value is `2018-01-01`

## Web Service

Web service plugs into the API and exposes the ability to edit the tags and the start date

users can seperate tags any way they want as the javascript will handle pathing, currently we support `[',', '\\', '/', '.', ' ', '\t']`

the datepicker will santize the input before pathing

