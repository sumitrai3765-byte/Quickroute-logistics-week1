"""
Phase 4: Predictive Modeling — Route Optimization (per zone)
------------------------------------------------------------
Solves a small Vehicle Routing Problem (VRP) within each delivery zone to
minimize total distance/time, reducing Average Cost per Delivery.

Requires: ortools (pip install ortools)

Input : data/orders_with_zones.csv
Output: printed optimized stop order per zone
"""

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

VEHICLES_PER_ZONE = 1  # single-vehicle sequencing per zone; raise for larger zones


def build_distance_matrix(coords: pd.DataFrame) -> np.ndarray:
    """Simple Euclidean distance matrix (swap for a road-network distance
    matrix, e.g. via OSMnx, for production use)."""
    n = len(coords)
    matrix = np.zeros((n, n))
    lat_lon = coords[["lat", "lon"]].to_numpy()
    for i in range(n):
        for j in range(n):
            matrix[i, j] = np.linalg.norm(lat_lon[i] - lat_lon[j])
    # Scale to integers (OR-Tools requires integer costs) — proxy "meters"
    return (matrix * 100_000).astype(int)


def solve_zone_route(coords: pd.DataFrame, num_vehicles: int = VEHICLES_PER_ZONE):
    distance_matrix = build_distance_matrix(coords)
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return None

    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))
    return route


def main():
    orders = pd.read_csv("data/orders_with_zones.csv")

    for zone_id, zone_orders in orders.groupby("zone_id"):
        coords = zone_orders[["lat", "lon"]].dropna().reset_index(drop=True)
        if len(coords) < 2:
            continue
        route = solve_zone_route(coords)
        print(f"Zone {zone_id}: optimized stop order (index) = {route}")


if __name__ == "__main__":
    main()
