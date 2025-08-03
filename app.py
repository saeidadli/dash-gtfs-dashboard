import os
import gtfs_kit as gk
from dash import Dash, html, dcc, Input, Output, callback, State, ALL
import dash_leaflet as dl
import plotly.express as px
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

print("Loading GTFS feed...")
feed = gk.read_feed("https://gtfs.at.govt.nz/gtfs.zip", dist_units="km")
feed = feed.clean()

week = feed.get_first_week()
trip_stats = feed.compute_trip_stats()
feed_stats = feed.compute_feed_stats(trip_stats, dates=[week[5]])
feed_time_series = feed.compute_feed_time_series(trip_stats, dates=[week[5]], freq='60Min')

colors = {
    0: "#9b59b6",   # LightRail
    1: "#34495e",   # Subway
    2: "#e74c3c",   # Rail
    3: "#27ae60",   # Bus
    4: "#3498db",   # Ferry
    5: "#f1c40f",   # CableTram
    6: "#1abc9c",   # AerialLift
    7: "#e67e22",   # Funicular
    11: "#95a5a6",  # Trolleybus
    12: "#8e44ad"   # Monorail
}
types = {
    0: "LightRail",
    1: "Subway",
    2: "Rail",
    3: "Bus",
    4: "Ferry",
    5: "CableTram",
    6: "AerialLift",
    7: "Funicular",
    11: "Trolleybus",
    12: "Monorail"
}

route_stats = feed.compute_route_stats(trip_stats, dates=[week[5]])
route_stats['route_color'] = route_stats['route_type'].map(colors)
route_stats['route_desc'] = route_stats['route_type'].map(types)

try:
    routes_gdf = feed.geometrize_routes()
    routes_gdf['route_color'] = routes_gdf['route_type'].map(colors)
    routes_gdf['route_desc'] = routes_gdf['route_type'].map(types)
    print(f"Loaded {len(routes_gdf)} route geometries")
except Exception as e:
    print(f"Could not load route geometries: {e}")
    routes_gdf = None

stops_gdf = feed.geometrize_stops()
stop_stats = feed.compute_stop_stats(dates=[week[5]])

stop_times = feed.stop_times.copy()
trips = feed.trips.copy()
stop_route_info = stop_times.merge(trips[['trip_id', 'route_id']], on='trip_id')
stop_route_info = stop_route_info.merge(feed.routes[['route_id', 'route_type']], on='route_id')

stop_primary_type = stop_route_info.groupby('stop_id')['route_type'].agg(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
).reset_index()
stop_primary_type.columns = ['stop_id', 'primary_route_type']

stops_gdf = stops_gdf.merge(stop_primary_type, on='stop_id', how='left')
stops_gdf['stop_color'] = stops_gdf['primary_route_type'].map(colors).fillna('#95a5a6')
stops_gdf['stop_desc'] = stops_gdf['primary_route_type'].map(types).fillna('Unknown')

am_peak_trips = trip_stats[(trip_stats['start_time'] >= '07:00:00') & (trip_stats['start_time'] <= '09:00:00') & (trip_stats['direction_id'] == 0)].copy()
route_frequencies = am_peak_trips.groupby('route_id')['trip_id'].count().reset_index(name='num_trips')
route_frequencies['headway'] = 120 / route_frequencies['num_trips'].replace(0, 1)
route_frequencies['headway_name'] = route_frequencies['headway'].apply(lambda x: "frequent" if x <= 15 else "connector" if x <= 30 else "local")

routes_gdf = routes_gdf.merge(
    route_frequencies[['route_id', 'headway_name']], 
    on='route_id', 
    how='left',
).fillna({'headway_name': 'unknown'})

print(f"Loaded {len(feed.routes)} routes and {len(stops_gdf)} stops")

app = Dash(__name__, suppress_callback_exceptions=True)

def create_map():
    map_children = [dl.TileLayer()]
    
    # Add routes in separate layers
    if routes_gdf is not None:
        for route_type, color in colors.items():
            type_routes = routes_gdf[routes_gdf['route_type'] == route_type].copy()
            for headway in ['frequent', 'connector', 'local']:
                headway_routes = type_routes[type_routes['headway_name'] == headway].copy()
                if len(headway_routes) > 0:
                    features = []
                    weight = 6 if headway == 'frequent' else 3 if headway == 'connector' else 1
                    for _, route in headway_routes.iterrows():
                        if hasattr(route.geometry, '__geo_interface__'):
                            features.append({
                                "type": "Feature",
                                "properties": {
                                    "route_id": str(route['route_id']),
                                    "name": str(route.get('route_short_name', route['route_id'])),
                                    "type": route['route_desc'],
                                },
                                "geometry": route.geometry.__geo_interface__
                            })
                
                    if features:
                        map_children.append(
                            dl.GeoJSON(
                                data={"type": "FeatureCollection", "features": features},
                                id={"type": "route-layer", "index": f"{headway}-{route_type}"},
                                options={"style": {"color": color, "opacity": 0.8, "weight": weight}},
                                hoverStyle={"color": "#f39c12", "weight": weight + 2, "opacity": 1},
                            )
                        )
    
    # Add empty stops layer
    map_children.append(dl.GeoJSON(
        id="stops-layer", 
        data={"type": "FeatureCollection", "features": []}
    ))
    
    # Add info display
    map_children.append(html.Div(id="info-display", style={
        'position': 'absolute', 'top': '10px', 'right': '10px', 'z-index': '1000',
        'background': 'rgba(255,255,255,0.95)', 'padding': '15px', 'border-radius': '10px',
        'box-shadow': '0 4px 15px rgba(0,0,0,0.2)', 'max-width': '300px', 'font-size': '14px'
    }))
    
    return dl.Map(
        id="map",
        children=map_children,
        style={'width': '100%', 'height': '70vh', 'position': 'relative', 'border-radius': '15px', 'overflow': 'hidden'},
        center=[-36.8485, 174.7633],
        zoom=10
    )

@callback(
    Output("stops-layer", "data"),
    [Input("map", "zoom"), Input("map", "bounds")]
)
def update_stops(zoom, bounds):
    if zoom is None or zoom < 13 or bounds is None:
        return {"type": "FeatureCollection", "features": []}
    
    # Filter stops within bounds
    lat_min, lat_max = bounds[0][0], bounds[1][0]
    lon_min, lon_max = bounds[0][1], bounds[1][1]
    
    visible_stops = stops_gdf[
        (stops_gdf.geometry.y >= lat_min) & (stops_gdf.geometry.y <= lat_max) &
        (stops_gdf.geometry.x >= lon_min) & (stops_gdf.geometry.x <= lon_max)
    ].head(200)
    
    features = []
    for _, stop in visible_stops.iterrows():
        features.append({
            "type": "Feature",
            "properties": {
                "stop_id": str(stop['stop_id']),
                "name": str(stop.get('stop_name', stop['stop_id'])),
                "type": stop['stop_desc']
            },
            "geometry": Point(stop.geometry.x, stop.geometry.y).__geo_interface__
        })
    
    return {"type": "FeatureCollection", "features": features}

@callback(
    Output("info-display", "children"),
    [Input("stops-layer", "clickData"), Input({"type": "route-layer", "index": ALL}, "clickData")]
)
def display_info(stop_click, route_clicks):
    # Check stops click
    if stop_click:
        stop_id = stop_click['properties']['stop_id']
        stop_stat = stop_stats[stop_stats['stop_id'] == stop_id]
        if not stop_stat.empty:
            stat = stop_stat.iloc[0]
            return html.Div([
                html.H4(f"ðﾟﾚﾏ {stop_click['properties']['name']}", style={'margin': '0 0 10px 0', 'color': '#2c3e50'}),
                html.P(f"Type: {stop_click['properties']['type']}", style={'margin': '5px 0'}),
                html.P(f"Routes: {stat['num_routes']}", style={'margin': '5px 0'}),
                html.P(f"Trips: {stat['num_trips']}", style={'margin': '5px 0'}),
                html.P(f"Hours: {stat['start_time']} - {stat['end_time']}", style={'margin': '5px 0'}),
                html.P(f"Headway: {stat['mean_headway']:.1f} min avg", style={'margin': '5px 0'})
            ])
    
    # Check routes click
    for route_click in route_clicks:
        if route_click:
            route_id = route_click['properties']['route_id']
            route_stat = route_stats[route_stats['route_id'] == route_id]
            if not route_stat.empty:
                stat = route_stat.iloc[0]
                return html.Div([
                    html.H4(f"ðﾟﾚﾌ Route {stat['route_short_name']}", style={'margin': '0 0 10px 0', 'color': '#2c3e50'}),
                    html.P(f"Type: {stat['route_desc']}", style={'margin': '5px 0'}),
                    html.P(f"Trips: {stat['num_trips']}", style={'margin': '5px 0'}),
                    html.P(f"Hours: {stat['start_time']} - {stat['end_time']}", style={'margin': '5px 0'}),
                    html.P(f"Distance: {stat['service_distance']:.1f} km", style={'margin': '5px 0'}),
                    html.P(f"Speed: {stat['service_speed']:.1f} km/h", style={'margin': '5px 0'})
                ])
    
    return html.P("Click on a route or stop for details", style={'color': '#7f8c8d', 'font-style': 'italic'})

def create_top_routes_chart():
    route_trip_counts = stop_route_info.groupby('route_id').size().reset_index(name='trip_count')
    routes_df = feed.routes
    route_trip_counts = route_trip_counts.merge(routes_df[['route_id', 'route_short_name', 'route_type']], on='route_id')
    route_trip_counts['route_desc'] = route_trip_counts['route_type'].map(types)
    top_routes = route_trip_counts.nlargest(8, 'trip_count')
    
    fig = px.bar(
        top_routes, x='trip_count', y='route_short_name',
        title="Top Busiest Routes", color='route_desc',
        orientation='h'
    )
    fig.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(title_text='Number of Trips')
    fig.update_yaxes(title_text='Route Name')
    fig.update_layout(barmode='group', xaxis_tickangle=-45)
    return fig

def create_top_stops_chart():
    stop_trip_counts = stop_route_info.groupby('stop_id').size().reset_index(name='trip_count')
    stop_trip_counts = stop_trip_counts.merge(stops_gdf[['stop_id', 'stop_name', 'stop_desc']], on='stop_id')
    top_stops = stop_trip_counts.nlargest(8, 'trip_count')
    top_stops['short_name'] = top_stops['stop_name'].str[:25] + '...'
    
    fig = px.bar(
        top_stops, x='trip_count', y='short_name',
        title="Top Busiest Stops", color='stop_desc',
        orientation='h'
    )
    fig.update_layout(height=350, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(title_text='Number of Trips')
    fig.update_yaxes(title_text='Stop Name')
    fig.update_layout(barmode='group', xaxis_tickangle=-45)
    return fig

def create_trip_chart():
    fig = px.bar(
        feed_time_series, x=feed_time_series.index, y='num_trips',
        title="Active Trips Throughout Day"
    )
    fig.update_layout(height=300, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(title_text='Time of Day')
    fig.update_yaxes(title_text='Number of Active Trips')

    return fig

app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🚌 Auckland Public Transport Dashboard", 
                style={'color': 'white', 'margin': '0', 'font-size': '2.5rem', 'text-shadow': '2px 2px 4px rgba(0,0,0,0.3)'})
    ], style={
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'padding': '30px', 'text-align': 'center', 'margin-bottom': '30px'
    }),
    
    # Stats cards
    html.Div([
        html.Div([
            html.Div("🚌", style={'font-size': '2rem', 'margin-bottom': '10px'}),
            html.H2(f"{feed_stats['num_routes'][0]}", style={'margin': '0', 'color': '#2c3e50', 'font-size': '2.5rem'}),
            html.P("Routes", style={'margin': '5px 0', 'color': '#7f8c8d', 'font-weight': 'bold'})
        ], className='stat-card'),
        
        html.Div([
            html.Div("⏱️", style={'font-size': '2rem', 'margin-bottom': '10px'}),
            html.H2(f"{feed_stats['service_duration'][0]/60:.1f}", style={'margin': '0', 'color': '#2c3e50', 'font-size': '2.5rem'}),
            html.P("Service Hours", style={'margin': '5px 0', 'color': '#7f8c8d', 'font-weight': 'bold'})
        ], className='stat-card'),
        
        html.Div([
            html.Div("📏", style={'font-size': '2rem', 'margin-bottom': '10px'}),
            html.H2(f"{feed_stats['service_distance'][0]/1000:.0f}", style={'margin': '0', 'color': '#2c3e50', 'font-size': '2.5rem'}),
            html.P("Distance (km)", style={'margin': '5px 0', 'color': '#7f8c8d', 'font-weight': 'bold'})
        ], className='stat-card'),
        
        html.Div([
            html.Div("🚏", style={'font-size': '2rem', 'margin-bottom': '10px'}),
            html.H2(f"{feed_stats['num_stops'][0]}", style={'margin': '0', 'color': '#2c3e50', 'font-size': '2.5rem'}),
            html.P("Stops", style={'margin': '5px 0', 'color': '#7f8c8d', 'font-weight': 'bold'})
        ], className='stat-card'),
        
        html.Div([
            html.Div("🚊", style={'font-size': '2rem', 'margin-bottom': '10px'}),
            html.H2(f"{feed_stats['num_trips'][0]}", style={'margin': '0', 'color': '#2c3e50', 'font-size': '2.5rem'}),
            html.P("Daily Trips", style={'margin': '5px 0', 'color': '#7f8c8d', 'font-weight': 'bold'})
        ], className='stat-card')
    ], style={'display': 'flex', 'justify-content': 'center', 'gap': '20px', 'margin-bottom': '40px', 'flex-wrap': 'wrap'}),
    
    # Map section
    html.Div([
        html.H2("🗺️ Interactive Transit Map", style={'color': '#2c3e50', 'margin-bottom': '20px', 'text-align': 'center'}),
        html.P("Click routes or stops for details. Stops appear when zoomed in.", 
               style={'text-align': 'center', 'color': '#7f8c8d', 'margin-bottom': '20px'}),
        create_map()
    ], style={'margin': '0 20px 40px 20px'}),
    
    # Charts section
    html.Div([
        # Trip timeline
        html.Div([
            dcc.Graph(figure=create_trip_chart())
        ], style={'background': 'white', 'border-radius': '15px', 'padding': '20px', 'box-shadow': '0 4px 15px rgba(0,0,0,0.1)', 'margin-bottom': '30px'}),
        
        # Side by side charts
        html.Div([
            html.Div([
                dcc.Graph(figure=create_top_routes_chart())
            ], style={'width': '48%', 'background': 'white', 'border-radius': '15px', 'padding': '20px', 'box-shadow': '0 4px 15px rgba(0,0,0,0.1)'}),
            
            html.Div([
                dcc.Graph(figure=create_top_stops_chart())
            ], style={'width': '48%', 'background': 'white', 'border-radius': '15px', 'padding': '20px', 'box-shadow': '0 4px 15px rgba(0,0,0,0.1)'})
        ], style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '30px'}),
        
        # Route types chart
        # html.Div([
        #     dcc.Graph(
        #         figure=px.bar(
        #             route_stats.groupby('route_desc').size().reset_index(name='count'),
        #             x='route_desc', y='count',
        #             title='Routes by Type'
        #         ).update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        #     )
        # ], style={'background': 'white', 'border-radius': '15px', 'padding': '20px', 'box-shadow': '0 4px 15px rgba(0,0,0,0.1)'})
    ], style={'margin': '0 20px'})
], style={
    'background': 'linear-gradient(to bottom, #f8f9fa 0%, #e9ecef 100%)',
    'min-height': '100vh', 'font-family': 'Arial, sans-serif'
})

# Add CSS for stat cards
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .stat-card {
                background: white;
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                min-width: 150px;
                transition: transform 0.3s ease;
            }
            .stat-card:hover {
                transform: translateY(-5px);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)
