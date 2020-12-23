---
layout: post
title: Northern China 2020 PM2.5 Pollution
subtitle: An animated visualization made with Geodata
tags: [China, Pollution]
comments: true

---

Plotting PM2.5 on a region over time. This is one of the methods I wrote for the Geodata package that Professor Michael R. Davidson is making at the University of California, San Deigo. Data came from NASA.


## Geodata Visualization

We are building the Geodata package to streamline the manipulation of renewable energy resource profiles and land use datasets with high geographical resolution. My work for this fall quarter 2020 was to creatw interactive charts and animations for geospatial masks and help generate renewable energy profiles based on land characteristics.

One of the visualization task I had, was that given a specific geographical region that the user is interested in, plot the PM2.5 distribution on a animated heatmap over time. Using the API call function from Geodata, I am able to collect the PM2.5 data of northern China of every hour in 2020 from Janurary to October. Then, we aggregate the PM2.5 over weeks and calculate the average for each week.

PM2.5 pollution data is taken from [MERRA-2 hourly surface flux diagnostics](https://disc.gsfc.nasa.gov/datasets/M2T1NXFLX_5.12.4/summary).

The visualization can be found here: [Link](https://mdavidson.org/geodata-viz/2020-12-07-northern-china-2020-pm25-pollution/#pm25-fine-particulate-matter-pollution-over-northern-china-in-2020)


### Motive of this sample animation

Here are some interesting reads provided by Prof. Davidson:
