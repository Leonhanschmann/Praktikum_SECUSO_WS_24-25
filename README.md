# Gaze Tracking Applications

This repository contains two Python applications that leverage eye-tracking technology with a Tobii eye tracker for biometric authentication. Developed during the Praktikum "Security, Usability, and Society" (Winter 2024/25), these applications collect and analyze unique gaze patterns as behavioral biometric identifiers. The GazeImageViewer authenticates users based on their gaze patterns while viewing images, while the Dot Task Gaze Tracker verifies identity through distinctive eye movement patterns during target-following tasks.

- [**GazeImageViewer**](./imagetask/): An image-based gaze tracking application that displays a sequence of images and collects real-time gaze data.
- [**Dot Task Gaze Tracker**](./dottask/): An application that conducts the dot task for gaze tracking, analyzing fixations and saccades, and providing comprehensive (and interactive) visualizations.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

### GazeImageViewer

GazeImageViewer is a Python application that presents a series of images to the user while collecting gaze data using a Tobii eye tracker. It provides real-time gaze visualization, heatmap generation.

For detailed information, please refer to the [GazeImageViewer README](./imagetask/README.md).

### Dot Task Gaze Tracker

The Dot Task Gaze Tracker is an application designed to conduct the dot task, collecting gaze data to analyze fixations and saccades. It provides comprehensive offline analysis, generating visualizations like gaze paths, heatmaps, fixations, saccades, and calculates analytical metrics with options for filtering and highlighting.

For detailed information, please refer to the [Dot Task Gaze Tracker README](./dottask/README.md).

---

## Features

### Common Features

- **Real-time Gaze Tracking**: Collect gaze data using a Tobii eye tracker.
- **Gaze Visualization**: Visualize gaze points, paths, fixations, and saccades.
- **Heatmap Generation**: Generate heatmaps to represent gaze intensity.
- **User Interaction**: Interactive controls for navigating and analyzing data.
- **Modular Design**: Clean codebase with separation between processing and visualization components.

### GazeImageViewer Features

- **Image Sequence Display**: Presents a series of images for a specified duration.
- **Real-time Gaze Overlay**: Displays gaze trails and glow effects on images.
- **Heatmap Overlays**: Generates heatmaps over images to highlight areas of interest.
- **Keyboard Controls**: Navigate through images and heatmaps.

### Dot Task Gaze Tracker Features

- **Dot Task Execution**: Conducts the dot task with a series of targets for gaze fixation.
- **Gaze Data Processing**: Offline analysis to detect fixations and saccades.
- **Comprehensive Metrics**: Calculates fixation durations, saccade amplitudes, velocities, and more.
- **Interactive Visualizations**: Includes gaze paths, fixations, saccades, heatmaps, and metrics summaries.
- **Filtering and Highlighting**: Offers options to filter and highlight gaze events.

---

## Project Structure

```
.
├── dottask/
│   ├── processors/
│   ├── views/
│   ├── images/
│   ├── main.py
│   └── README.md
├── imagetask/
│   ├── processors/
│   ├── views/
│   ├── images/
│   ├── main.py
│   └── README.md
├── videos/
│   ├── dottask_heatmap.mp4
│   ├── dottask.mp4
│   └── imagetask.mp4
└── README.md
```

- **dottask/**: Contains the Dot Task Gaze Tracker application.
- **imagetask/**: Contains the GazeImageViewer application.
- **README.md**: Main README file (this file).

---

## Installation

Each application contains detailed installation instructions in their respective README files. Please navigate to the applications' directories for specific installation steps.

- [GazeImageViewer Installation Instructions](./imagetask/README.md#installation)
- [Dot Task Gaze Tracker Installation Instructions](./dottask/README.md#installation)

---

## Usage

Instructions on how to run and use each application are provided in their respective README files.

- [GazeImageViewer Usage Instructions](./imagetask/README.md#usage)
- [Dot Task Gaze Tracker Usage Instructions](./dottask/README.md#usage)

---

## Demo Videos

The repository includes demonstration videos showcasing the functionality of both applications:

- **dottask.mp4**: Demonstrates the basic functionality of the Dot Task Gaze Tracker application.
- **dottask_heatmap.mp4**: Shows the heatmap generation and analysis features of the Dot Task application.
- **imagetask.mp4**: Demonstrates the GazeImageViewer application in action, including gaze tracking,  image navigation and the heatmap visualization.


