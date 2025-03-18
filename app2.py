import streamlit as st
import re
import pandas as pd
import numpy as np
import time
import folium
from streamlit_folium import folium_static
import random
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

st.set_page_config(page_title="Vehicle Routing Problem Solver", layout="wide")

st.title("Vehicle Routing Problem with Pickup and Delivery")
st.write("""
 This app helps you solve vehicle routing problems with pickup and delivery constraints.
 Enter the locations, demands, vehicle capacities, and pickup-delivery pairs to find the optimal routes.
 """)

# Initialize session state variables if they don't exist
if 'addresses' not in st.session_state:
    st.session_state.addresses = []
if 'demands' not in st.session_state:
    st.session_state.demands = []
if 'vehicle_capacities' not in st.session_state:
    st.session_state.vehicle_capacities = []
if 'pickups_deliveries' not in st.session_state:
    st.session_state.pickups_deliveries = []
if 'distance_matrix' not in st.session_state:
    st.session_state.distance_matrix = []
if 'locations' not in st.session_state:
    st.session_state.locations = []
if 'solution_found' not in st.session_state:
    st.session_state.solution_found = False
if 'solution_details' not in st.session_state:
    st.session_state.solution_details = {}


# Function to geocode addresses
@st.cache_data
def geocode_address(address):
    geolocator = Nominatim(user_agent="streamlit_vrp_app_v1")
    try:
        # Respect Nominatim's rate limit
        time.sleep(1)
        location = geolocator.geocode(address)
        if location:
            return (location.latitude, location.longitude)
        else:
            return None
    except Exception as e:
        st.error(f"Geocoding error for '{address}': {str(e)}")
        return None


# Function to calculate distance matrix
def calculate_distance_matrix(locations):
    num_locations = len(locations)
    distance_matrix = []
    for i in range(num_locations):
        row = []
        for j in range(num_locations):
            if i == j:
                row.append(0)
            elif (i == 0 and j == 1) or (i == 1 and j == 0):
                # Distance between depot and its duplicate is 0
                row.append(0)
            else:
                # Calculate distance in kilometers
                distance = round(geodesic(locations[i], locations[j]).km, 2)
                row.append(distance)
        distance_matrix.append(row)
    return distance_matrix


# Function to solve the VRP
def solve_vrp(data):
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # Convert the float distance to an integer (multiply by 100 to preserve 2 decimal places)
        return int(data["distance_matrix"][from_node][to_node] * 100)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        300000,  # vehicle maximum travel distance (3000 km * 100)
        True,  # start cumul to zero
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Add Capacity constraint
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Define Transportation Requests
    for request in data["pickups_deliveries"]:
        pickup_index = manager.NodeToIndex(request[0])
        delivery_index = manager.NodeToIndex(request[1])
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        routing.solver().Add(
            routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
        )
        routing.solver().Add(
            distance_dimension.CumulVar(pickup_index)
            <= distance_dimension.CumulVar(delivery_index)
        )

    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Process the solution
    if solution:
        result = {
            "objective_value": solution.ObjectiveValue(),
            "routes": [],
            "total_distance": 0,
            "total_load": 0
        }

        for vehicle_id in range(data["num_vehicles"]):
            if not routing.IsVehicleUsed(solution, vehicle_id):
                continue

            index = routing.Start(vehicle_id)
            route_info = {
                "vehicle_id": vehicle_id,
                "route": [],
                "distance": 0,
                "load": 0
            }

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_info["load"] += data["demands"][node_index]
                route_info["route"].append({
                    "node": node_index,
                    "address": data["addresses"][node_index],
                    "load": route_info["load"],
                    "location": data["locations"][node_index] if "locations" in data else None
                })

                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_info["distance"] += routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )

            # Add the final depot
            node_index = manager.IndexToNode(index)
            route_info["route"].append({
                "node": node_index,
                "address": data["addresses"][node_index],
                "load": route_info["load"],
                "location": data["locations"][node_index] if "locations" in data else None
            })

            # Convert distance back to kilometers (divide by 100)
            route_info["distance"] /= 100

            result["routes"].append(route_info)
            result["total_distance"] += route_info["distance"]
            result["total_load"] += route_info["load"]

        return result
    else:
        return None


# Function to create a map with route visualizations
def create_route_map(locations, routes):
    # Generate random colors for each vehicle route
    def random_color():
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return f'#{r:02x}{g:02x}{b:02x}'

    # Get colors for each vehicle
    vehicle_colors = [random_color() for _ in range(len(routes))]

    # Start with the first location's coordinates for the map center
    if locations:
        center_lat = sum(loc[0] for loc in locations) / len(locations)
        center_lng = sum(loc[1] for loc in locations) / len(locations)
    else:
        # Default center if no locations
        center_lat, center_lng = 0, 0

    # Create a map
    m = folium.Map(location=[center_lat, center_lng], zoom_start=10)

    # Add depot marker
    if locations:
        folium.Marker(
            locations[0],
            popup="Depot",
            tooltip="Depot",
            icon=folium.Icon(color='red', icon='home')
        ).add_to(m)

    # Add routes for each vehicle
    for i, route in enumerate(routes):
        route_color = vehicle_colors[i]
        route_locations = []

        # Extract location data
        for stop in route['route']:
            if stop['location']:
                route_locations.append(stop['location'])

                # Determine icon color and type based on node type
                icon_color = 'blue'
                icon_type = 'info-sign'
                popup_text = f"Location {stop['node'] - 1}"

                # Check if pickup or delivery
                if stop['node'] > 1:  # Skip depot
                    demand = st.session_state.demands[stop['node']]
                    if demand > 0:
                        icon_color = 'green'
                        icon_type = 'arrow-up'
                        popup_text += " (Pickup)"
                    elif demand < 0:
                        icon_color = 'orange'
                        icon_type = 'arrow-down'
                        popup_text += " (Delivery)"

                # Skip adding another marker for depot if it's already added
                if stop['node'] > 1:
                    folium.Marker(
                        stop['location'],
                        popup=popup_text,
                        tooltip=f"Stop {stop['node']}",
                        icon=folium.Icon(color=icon_color, icon=icon_type)
                    ).add_to(m)

        # Add polyline for the route
        if len(route_locations) >= 2:
            folium.PolyLine(
                route_locations,
                color=route_color,
                weight=5,
                opacity=0.7,
                popup=f"Vehicle {route['vehicle_id'] + 1}: {route['distance']:.2f} km"
            ).add_to(m)

    return m


# Main app
with st.sidebar:
    st.header("Input Parameters")

    # Number of locations
    num_actual_locations = st.number_input("Number of locations (including depot)", min_value=2, value=3, step=1)

    # Number of vehicles
    num_vehicles = st.number_input("Number of vehicles", min_value=1, value=2, step=1)

    # Reset button
    if st.button("Reset All Data"):
        st.session_state.addresses = []
        st.session_state.demands = []
        st.session_state.vehicle_capacities = []
        st.session_state.pickups_deliveries = []
        st.session_state.distance_matrix = []
        st.session_state.locations = []
        st.session_state.solution_found = False
        st.session_state.solution_details = {}
        st.experimental_rerun()

# Main content
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Locations", "Demands", "Vehicles", "Pickup-Delivery", "Solution", "Map"])

# Tab 1: Locations
with tab1:
    st.header("Location Addresses")

    # Calculate the actual number of locations (including depot duplicate)
    num_locations = num_actual_locations + 1

    # Resize the addresses list if needed
    if len(st.session_state.addresses) != num_locations:
        # Keep existing addresses if available
        if len(st.session_state.addresses) > 0:
            depot_address = st.session_state.addresses[0]
            old_addresses = st.session_state.addresses[2:] if len(st.session_state.addresses) > 2 else []

            st.session_state.addresses = [depot_address, depot_address]
            st.session_state.addresses.extend(old_addresses)

            # Fill remaining slots with empty strings
            while len(st.session_state.addresses) < num_locations:
                st.session_state.addresses.append("")

            # Trim if too many
            if len(st.session_state.addresses) > num_locations:
                st.session_state.addresses = st.session_state.addresses[:num_locations]
        else:
            # Initialize with empty strings
            st.session_state.addresses = [""] * num_locations
            # Make sure the first two addresses are the same (depot and its duplicate)
            st.session_state.addresses[1] = st.session_state.addresses[0]

    # Depot address
    depot_address = st.text_input("Depot Address", key="depot_address", value=st.session_state.addresses[0])
    st.session_state.addresses[0] = depot_address
    st.session_state.addresses[1] = depot_address  # Duplicate for index 1

    # Other location addresses
    for i in range(2, num_locations):
        location_address = st.text_input(f"Location {i - 1} Address", key=f"location_{i}",
                                         value=st.session_state.addresses[i] if i < len(
                                             st.session_state.addresses) else "")
        if i < len(st.session_state.addresses):
            st.session_state.addresses[i] = location_address
        else:
            st.session_state.addresses.append(location_address)

    # Button to geocode addresses
    if st.button("Geocode Addresses and Calculate Distance Matrix"):
        with st.spinner("Geocoding addresses..."):
            # Reset locations list
            st.session_state.locations = []

            # Geocode each unique address
            unique_addresses = set(st.session_state.addresses)
            address_to_coordinates = {}

            for address in unique_addresses:
                if address:  # Skip empty addresses
                    coordinates = geocode_address(address)
                    if coordinates:
                        address_to_coordinates[address] = coordinates
                        st.success(f"Geocoded '{address}' ➔ ({coordinates[0]}, {coordinates[1]})")
                    else:
                        st.error(f"Could not geocode address: '{address}'")

            # Check if all addresses were geocoded
            if len(address_to_coordinates) == len(unique_addresses) and all(
                    address for address in st.session_state.addresses):
                # Map each address to its coordinates
                for address in st.session_state.addresses:
                    st.session_state.locations.append(address_to_coordinates[address])

                # Calculate the distance matrix
                st.session_state.distance_matrix = calculate_distance_matrix(st.session_state.locations)

                # Display the distance matrix
                st.subheader("Distance Matrix (km)")

                # Create a DataFrame for better visualization
                df_distance = pd.DataFrame(
                    st.session_state.distance_matrix,
                    index=[f"Location {i}" if i > 0 else "Depot" for i in range(len(st.session_state.distance_matrix))],
                    columns=[f"Location {i}" if i > 0 else "Depot" for i in
                             range(len(st.session_state.distance_matrix))]
                )
                st.dataframe(df_distance)
            else:
                st.error("Please enter valid addresses for all locations.")

# Tab 2: Demands
with tab2:
    st.header("Demands at Each Location")

    # Resize the demands list if needed
    if len(st.session_state.demands) != num_locations:
        # Keep existing demands if available
        temp_demands = [0]  # Depot (index 0) always has 0 demand

        if len(st.session_state.demands) > 1:
            depot_demand = st.session_state.demands[1]
            old_demands = st.session_state.demands[2:] if len(st.session_state.demands) > 2 else []

            temp_demands.append(depot_demand)
            temp_demands.extend(old_demands)
        else:
            temp_demands.append(0)  # Default demand for depot duplicate

        # Fill remaining slots with zeros
        while len(temp_demands) < num_locations:
            temp_demands.append(0)

        # Trim if too many
        if len(temp_demands) > num_locations:
            temp_demands = temp_demands[:num_locations]

        st.session_state.demands = temp_demands

    # Info message
    st.info("Enter positive values for pickup demands and negative values for delivery demands.")

    # Depot demand
    depot_demand = st.number_input("Demand at Depot Location", value=st.session_state.demands[1], step=1,
                                   key="depot_demand")
    st.session_state.demands[1] = depot_demand

    # Other location demands
    for i in range(2, num_locations):
        location_demand = st.number_input(
            f"Demand at Location {i - 1}",
            value=st.session_state.demands[i] if i < len(st.session_state.demands) else 0,
            step=1,
            key=f"demand_{i}"
        )
        if i < len(st.session_state.demands):
            st.session_state.demands[i] = location_demand
        else:
            st.session_state.demands.append(location_demand)

    # Display the demands
    if st.button("Show Demands Summary"):
        st.subheader("Demands Summary")

        # Create a DataFrame for better visualization
        df_demands = pd.DataFrame({
            "Location": ["Depot"] + [f"Location {i - 1}" for i in range(2, num_locations)],
            "Demand": [st.session_state.demands[1]] + [st.session_state.demands[i] for i in range(2, num_locations)],
            "Type": ["Depot"] + [
                "Pickup" if st.session_state.demands[i] > 0 else "Delivery" if st.session_state.demands[
                                                                                   i] < 0 else "Neutral" for i in
                range(2, num_locations)]
        })
        st.dataframe(df_demands)

# Tab 3: Vehicle Capacities
with tab3:
    st.header("Vehicle Capacities")

    # Resize the vehicle capacities list if needed
    if len(st.session_state.vehicle_capacities) != num_vehicles:
        # Keep existing capacities if available
        if len(st.session_state.vehicle_capacities) > 0:
            old_capacities = st.session_state.vehicle_capacities

            # Fill remaining slots with the last capacity value or 15 if empty
            default_capacity = old_capacities[-1] if old_capacities else 15

            while len(old_capacities) < num_vehicles:
                old_capacities.append(default_capacity)

            # Trim if too many
            if len(old_capacities) > num_vehicles:
                old_capacities = old_capacities[:num_vehicles]

            st.session_state.vehicle_capacities = old_capacities
        else:
            # Initialize with default capacity (15)
            st.session_state.vehicle_capacities = [15] * num_vehicles

    # Vehicle capacities input
    for i in range(num_vehicles):
        capacity = st.number_input(
            f"Capacity for Vehicle {i + 1}",
            min_value=1,
            value=st.session_state.vehicle_capacities[i] if i < len(st.session_state.vehicle_capacities) else 15,
            step=1,
            key=f"capacity_{i}"
        )
        if i < len(st.session_state.vehicle_capacities):
            st.session_state.vehicle_capacities[i] = capacity
        else:
            st.session_state.vehicle_capacities.append(capacity)

    # Display the vehicle capacities
    if st.button("Show Vehicle Capacities Summary"):
        st.subheader("Vehicle Capacities Summary")

        # Create a DataFrame for better visualization
        df_capacities = pd.DataFrame({
            "Vehicle": [f"Vehicle {i + 1}" for i in range(num_vehicles)],
            "Capacity": st.session_state.vehicle_capacities
        })
        st.dataframe(df_capacities)

# Tab 4: Pickup-Delivery Pairs
with tab4:
    st.header("Pickup-Delivery Pairs")

    # Number of pickup-delivery pairs
    num_pairs = st.number_input("Number of Pickup-Delivery Pairs", min_value=0,
                                value=len(st.session_state.pickups_deliveries), step=1)

    # Resize the pickups_deliveries list if needed
    if len(st.session_state.pickups_deliveries) != num_pairs:
        # Keep existing pairs if available
        old_pairs = st.session_state.pickups_deliveries.copy() if st.session_state.pickups_deliveries else []

        # Fill remaining slots with default values (depot and first location)
        while len(old_pairs) < num_pairs:
            old_pairs.append([1, 2])

        # Trim if too many
        if len(old_pairs) > num_pairs:
            old_pairs = old_pairs[:num_pairs]

        st.session_state.pickups_deliveries = old_pairs

    # Pickup-delivery pairs input
    for i in range(num_pairs):
        st.subheader(f"Pickup-Delivery Pair {i + 1}")

        col1, col2 = st.columns(2)

        with col1:
            pickup = st.selectbox(
                "Pickup Location",
                range(1, num_locations),
                index=st.session_state.pickups_deliveries[i][0] - 1 if i < len(
                    st.session_state.pickups_deliveries) else 0,
                format_func=lambda x: f"Location {x - 1}" if x > 1 else "Depot",
                key=f"pickup_{i}"
            )

        with col2:
            delivery = st.selectbox(
                "Delivery Location",
                range(1, num_locations),
                index=st.session_state.pickups_deliveries[i][1] - 1 if i < len(
                    st.session_state.pickups_deliveries) else 1,
                format_func=lambda x: f"Location {x - 1}" if x > 1 else "Depot",
                key=f"delivery_{i}"
            )

        if i < len(st.session_state.pickups_deliveries):
            st.session_state.pickups_deliveries[i] = [pickup, delivery]
        else:
            st.session_state.pickups_deliveries.append([pickup, delivery])

    # Display the pickup-delivery pairs
    if st.button("Show Pickup-Delivery Pairs Summary"):
        st.subheader("Pickup-Delivery Pairs Summary")

        # Create a DataFrame for better visualization
        df_pairs = pd.DataFrame({
            "Pair": [f"Pair {i + 1}" for i in range(num_pairs)],
            "Pickup": [f"Location {p[0] - 1}" if p[0] > 1 else "Depot" for p in st.session_state.pickups_deliveries],
            "Delivery": [f"Location {p[1] - 1}" if p[1] > 1 else "Depot" for p in st.session_state.pickups_deliveries]
        })
        st.dataframe(df_pairs)

# Tab 5: Solution
with tab5:
    st.header("Solution")

    # Check if all inputs are valid
    inputs_valid = (
            len(st.session_state.addresses) == num_locations and
            len(st.session_state.demands) == num_locations and
            len(st.session_state.vehicle_capacities) == num_vehicles and
            len(st.session_state.distance_matrix) == num_locations
    )

    if not inputs_valid:
        st.warning("Please complete all input sections before solving.")
    else:
        # Prepare data for the solver
        data = {
            "distance_matrix": st.session_state.distance_matrix,
            "num_vehicles": num_vehicles,
            "depot": 0,
            "demands": st.session_state.demands,
            "vehicle_capacities": st.session_state.vehicle_capacities,
            "pickups_deliveries": st.session_state.pickups_deliveries,
            "addresses": st.session_state.addresses,
            "locations": st.session_state.locations  # Add locations for mapping
        }

        # Display the data structure for confirmation
        st.subheader("Data Structure for Confirmation")
        st.write("Please review the following data structure that will be used for solving:")

        # Format the distance matrix to be more readable
        formatted_distance_matrix = []
        for row in data["distance_matrix"]:
            formatted_distance_matrix.append([round(dist, 2) for dist in row])

        formatted_data = {
            "distance_matrix": formatted_distance_matrix,
            "num_vehicles": data["num_vehicles"],
            "depot": data["depot"],
            "demands": data["demands"],
            "vehicle_capacities": data["vehicle_capacities"],
            "pickups_deliveries": data["pickups_deliveries"]
        }

        # Create a collapsible section for the data structure
        with st.expander("View Data Dictionary Structure", expanded=True):
            # Display the formatted data structure
            st.code(f"data = {formatted_data}", language="python")

            # Add a more user-friendly explanation
            st.markdown("### Explanation:")
            st.markdown("""
             - **distance_matrix**: Distances between locations in km. Row and column 0 is the depot.
             - **num_vehicles**: Total number of vehicles available.
             - **depot**: Index of the depot (0).
             - **demands**: List of demands at each location (positive for pickup, negative for delivery).
             - **vehicle_capacities**: Maximum capacity of each vehicle.
             - **pickups_deliveries**: List of pairs where each pair contains [pickup_index, delivery_index].
             """)

        if st.button("Solve VRP"):
            with st.spinner("Solving Vehicle Routing Problem..."):
                # Prepare data for the solver
                data = {
                    "distance_matrix": st.session_state.distance_matrix,
                    "num_vehicles": num_vehicles,
                    "depot": 0,
                    "demands": st.session_state.demands,
                    "vehicle_capacities": st.session_state.vehicle_capacities,
                    "pickups_deliveries": st.session_state.pickups_deliveries,
                    "addresses": st.session_state.addresses,
                    "locations": st.session_state.locations  # Add locations for mapping
                }

                # Solve the VRP
                solution = solve_vrp(data)

                if solution:
                    st.session_state.solution_found = True
                    st.session_state.solution_details = solution
                else:
                    st.session_state.solution_found = False
                    st.error("No solution found! Please check your inputs.")

    # Display solution if available
    if st.session_state.solution_found:
        st.success("Solution found!")

        st.subheader("Objective Value")
        st.write(f"Total objective value: {st.session_state.solution_details['objective_value']}")

        st.subheader("Total Distance")
        st.write(f"Total distance of all routes: {st.session_state.solution_details['total_distance']:.2f} km")

        st.subheader("Total Load")
        st.write(f"Total load of all routes: {st.session_state.solution_details['total_load']}")

        st.subheader("Routes")
        for route in st.session_state.solution_details['routes']:
            st.write(f"### Route for Vehicle {route['vehicle_id'] + 1}")

            # Create a DataFrame for the route
            route_data = []
            for stop in route['route']:
                location_name = "Depot" if stop['node'] <= 1 else f"Location {stop['node'] - 1}"
                route_data.append({
                    "Stop": location_name,
                    "Address": stop['address'],
                    "Load": stop['load']
                })

            df_route = pd.DataFrame(route_data)
            st.dataframe(df_route)

            st.write(f"Distance of the route: {route['distance']:.2f} km")
            st.write(f"Final load of the route: {route['load']}")
            st.write("---")

# Tab 6: Map Visualization
with tab6:
    st.header("Route Map Visualization")

    if not st.session_state.solution_found:
        st.warning("Please solve the VRP problem first to see the route visualization.")
    elif not st.session_state.locations:
        st.warning("No geocoded locations available. Please geocode addresses in the Locations tab.")
    else:
        st.write("The map below shows the optimized routes for each vehicle:")

        # Create the map
        route_map = create_route_map(
            st.session_state.locations,
            st.session_state.solution_details['routes']
        )

        # Display the map
        folium_static(route_map, width=1000, height=600)

        # Display legend
        st.subheader("Map Legend")
        legend_col1, legend_col2, legend_col3 = st.columns(3)

        with legend_col1:
            st.markdown("🔴 **Red**: Depot location")

        with legend_col2:
            st.markdown("🟢 **Green**: Pickup location")

        with legend_col3:
            st.markdown("🟠 **Orange**: Delivery location")

        # Add route statistics in expandable sections
        st.subheader("Route Details")

        for i, route in enumerate(st.session_state.solution_details['routes']):
            with st.expander(f"Vehicle {route['vehicle_id'] + 1} Route Details"):
                # Create two columns for route info
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total Distance", f"{route['distance']:.2f} km")

                with col2:
                    st.metric("Total Load", f"{route['load']}")

                # Create a table showing the sequence
                route_sequence = []
                for j, stop in enumerate(route['route']):
                    location_name = "Depot" if stop['node'] <= 1 else f"Location {stop['node'] - 1}"
                    stop_type = "Depot" if stop['node'] <= 1 else (
                        "Pickup" if st.session_state.demands[stop['node']] > 0 else
                        "Delivery" if st.session_state.demands[stop['node']] < 0 else "Stop"
                    )

                    # Calculate distance from previous stop if not the first stop
                    distance_from_prev = 0
                    if j > 0:
                        prev_location = route['route'][j - 1]['location']
                        curr_location = stop['location']
                        if prev_location and curr_location:
                            distance_from_prev = geodesic(prev_location, curr_location).km

                    route_sequence.append({
                        "Sequence": j + 1,
                        "Type": stop_type,
                        "Location": location_name,
                        "Address": stop['address'],
                        "Load After Stop": stop['load'],
                        "Distance from Previous (km)": f"{distance_from_prev:.2f}" if j > 0 else "N/A"
                    })

                # Show the sequence table
                st.table(pd.DataFrame(route_sequence))

        # Add export feature
        st.subheader("Export Results")


        # Create a download button for route details
        def generate_route_csv():
            all_stops = []
            for route in st.session_state.solution_details['routes']:
                vehicle_id = route['vehicle_id'] + 1
                for i, stop in enumerate(route['route']):
                    location_name = "Depot" if stop['node'] <= 1 else f"Location {stop['node'] - 1}"
                    stop_type = "Depot" if stop['node'] <= 1 else (
                        "Pickup" if st.session_state.demands[stop['node']] > 0 else
                        "Delivery" if st.session_state.demands[stop['node']] < 0 else "Stop"
                    )
                    all_stops.append({
                        "Vehicle": vehicle_id,
                        "Stop Sequence": i + 1,
                        "Location": location_name,
                        "Type": stop_type,
                        "Address": stop['address'],
                        "Load": stop['load'],
                        "Latitude": stop['location'][0] if stop['location'] else None,
                        "Longitude": stop['location'][1] if stop['location'] else None
                    })

            return pd.DataFrame(all_stops).to_csv(index=False)


        csv = generate_route_csv()
        st.download_button(
            label="Download Route Details as CSV",
            data=csv,
            file_name="vrp_route_details.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    pass
