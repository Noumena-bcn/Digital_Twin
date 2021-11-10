# Node Red

This directory contains the flows for communication between with the server

## Requirements

- Install Node Red

### Install Node Red Ubuntu

The command will add the NodeSource signing key to your system.

```
curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
```

Install Node.js and npm by typing:

```
sudo apt install nodejs
```

Check Node.js version

```
node --version
```

Check npm version

```
npm --version
```

Install Node Red

```
sudo npm install -g --unsafe-perm node-red
```

Start Node Red

```
node-red -v
```


## JSON_Occupancy

The JSON format that is created for occupancy has the following structure:

```
	{
	   "Zone": {
	   
	      "Zone1": (value),
	      "Zone2": (value),
	      "Zone7": (value)
	      
	   },
	   
	   "General": (value),
	   "Time": (str),
	   "Date": (str)
	},
```

The result it can be found on [JSON_Occupancy](https://github.com/Noumena-bcn/Digital_Twin/tree/main/src/JSON_Occupancy)

## Final_JSON

The JSON format that is created for DataModel has the following structure:

```
	{
	  "digital_twin": {
		    "temperature": {
		      "sensor1": (value),
		      "sensor2": (value),
		      "sensor3": (value),
		      "sensor4": (value),
		      "siemens": (value)
		    },
		    "huimidity": {
		      "sensor1": (value),
		      "sensor2": (value),
		      "sensor3": (value),
		      "sensor4": (value),
		      "siemens": (value)
		    },
		    "co2": {
		      "sensor1": (value),
		      "sensor2": (value),
		      "sensor3": (value),
		      "sensor4": (value)
		    },
		    "zone": {
		      "zone 1": (value),
		      "zone 2": (value),
		      "zone 3": (value),
		      "zone 4": (value),
		      "zone 5": (value),
		      "zone 6": (value),
		      "zone 7": (value)
		   }
		} 
	}
```


## Node Red Flow

That flow was created for the comunication with the server. Where the node "File IN" node is used to extract the data from the JSON file.
The data is extracted in string format, so using the "JSON" node, you can transform the string to JSON.
Then this information is send to the server using the "HTTP" node.

![Screenshot from 2021-06-23 12-04-22](https://user-images.githubusercontent.com/62296738/123079034-d6ed1800-d41b-11eb-9fcd-6fc23cbb54b6.png)


