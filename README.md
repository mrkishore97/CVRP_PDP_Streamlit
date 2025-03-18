# Vehicle Routing Problem (VRP) Solver

A Streamlit web application for solving vehicle routing problems with pickup and delivery constraints.

## Overview

This application helps logistics planners and supply chain managers solve complex vehicle routing problems. It allows users to:

- Define multiple locations (including a depot)
- Specify vehicle capacities
- Set pickup and delivery demands at each location
- Create pickup-delivery pairs
- Calculate optimal routes for a fleet of vehicles

## Features

- **Interactive UI**: Easy-to-use interface with tabbed organization for different parameters
- **Geocoding**: Converts addresses to coordinates using Nominatim geocoding service
- **Distance Matrix**: Automatically calculates distance matrix between locations
- **Optimization**: Uses Google OR-Tools for solving the routing problem
- **Visualization**: Displays routes and solution details in a readable format

## Installation

1. Clone this repository
2. Install required dependencies:

```bash
pip install streamlit pandas numpy geopy ortools
```

3. Run the application:

```bash
streamlit run vrp_solver.py
```

## Usage

1. **Locations**: Enter the depot address and other location addresses
2. **Demands**: Set the demand values for each location
   - Positive values for pickup demands
   - Negative values for delivery demands
3. **Vehicles**: Specify the number of vehicles and their capacities
4. **Pickup-Delivery Pairs**: Define the pickup and delivery location pairs
5. **Solution**: Click "Solve VRP" to find the optimal routes

## How It Works

This application uses the following workflow:

1. Geocode addresses to get latitude and longitude coordinates
2. Calculate distances between all locations using geodesic distance
3. Set up the vehicle routing problem with pickup and delivery constraints using OR-Tools
4. Solve the problem using the Parallel Cheapest Insertion algorithm
5. Display the solution, including routes for each vehicle, distances, and loads

## Scope for Improvement

- **Google Maps API Integration**: Replace the current Nominatim geocoding with Google Maps API for more accurate and reliable geocoding, faster response times, and better handling of address ambiguities
- **Route Visualization**: Add a map visualization of the routes
- **Time Windows**: Implement time window constraints for deliveries
- **Multiple Depots**: Add support for multiple depot locations
- **Real-time Traffic**: Incorporate real-time traffic data for more accurate route planning
- **Solution Comparison**: Add functionality to compare different solution strategies
- **Save/Load**: Allow saving and loading problem instances
- **Batch Processing**: Implement batch processing for large-scale problems

## Technical Details

- **Streamlit**: For the web interface
- **GeoPy**: For geocoding addresses
- **OR-Tools**: Google's optimization tools for solving the routing problem
- **Pandas**: For data manipulation and display

## License

[MIT License](LICENSE)

## Author

[Your Name]

## Acknowledgements

- This application uses Google's OR-Tools for solving the Vehicle Routing Problem
- Geocoding is provided by Nominatim through the GeoPy library
