import os
import threading
import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import logging
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

import plotly.io as pio
pio.templates.default = 'none'

import datetime

class Dashboard:
    def __init__(self, evaluation_render_dir):
        self.evaluation_render_dir = evaluation_render_dir
        # self.app = dash.Dash(__name__)
        self.app = dash.Dash(
                                __name__,
                                external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://use.fontawesome.com/releases/v5.8.1/css/all.css']
                            )

        self.setup_layout()
        self.setup_callbacks()
        self.server_thread = threading.Thread(target=self.run_server)
        self.server_thread.daemon = True

        # Adjust the logging level here
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

    def setup_layout(self):
        # Fetch available experiments and episodes
        experiments = self.get_available_experiments()
        default_experiment = experiments[-1] if experiments else None
        default_episodes = self.get_available_episodes(default_experiment) if default_experiment else []
        default_episode = default_episodes[-1] if default_episodes else None
    
        self.app.layout =  dbc.Container([
            html.H1("SustainDC - Data Center Sustainability Dashboard", className='text-center my-4', style={'font-size': '32px', 'font-weight': 'bold'}),

            # Keep the dropdown menus for selecting experiment and episode
            # Replace dcc.Dropdown with dbc.Select for experiment selection
            # Row for dropdowns and button
            # Row for dropdowns and button, centered and with smaller components
            dbc.Row([
                dbc.Col([
                    dbc.Label("Select Experiment", style={'font-size': '20px'}),
                    dbc.Select(
                        id='experiment-dropdown',
                        options=[{'label': exp, 'value': exp} for exp in experiments],
                        value=default_experiment,  # Set default experiment
                        placeholder="Select Experiment",
                        style={'width': '300px', 'font-size': '18px'},
                    ),
                ], width='auto', align='center'),
                dbc.Col([
                    dbc.Label("Select Rendered Episode", style={'font-size': '20px'}),
                    dbc.Select(
                        id='episode-dropdown',
                        options=[{'label': ep, 'value': ep} for ep in default_episodes],
                        value=default_episode,  # Set default episode
                        placeholder="Select Rendered Episode",
                        style={'width': '300px', 'font-size': '18px'},
                    ),
                ], width='auto', align='center'),
                dbc.Col([
                    dbc.Label(" "),  # Empty label to match the structure
                    dbc.Button(
                        [html.I(className='fas fa-pause me-2'), 'Pause'],
                        id='pause-button',
                        n_clicks=0,
                        color='primary',
                        style={'font-size': '18px'},  # Increase button font size
                    ),
                ], width='auto', align='center'),
                dbc.Col([
                    dbc.Label("Update Interval (ms)", style={'font-size': '20px'}),
                    dbc.Input(
                        id='interval-input',
                        type='number',
                        min=500,
                        max=10000,
                        step=500,
                        value=2000,
                        style={'width': '200px', 'font-size': '18px'},
                    ),
                ], width='auto', align='center'),
            ], justify='center', className='mb-4'),
            # Store component to hold the pause state
            dcc.Store(id='pause-state', data=False),
            # **First Section: Results with Vertical Label**
            html.Hr(),
            dbc.Row([
                dbc.Col(
                    html.Div("Results", style={
                        'writing-mode': 'vertical-rl',
                        'transform': 'rotate(180deg)',
                        'text-align': 'center',
                        'font-size': '28px',
                        'font-weight': 'bold',
                        'white-space': 'nowrap',  # Prevent text wrapping
                    }),
                    # Remove width parameter
                    # width=1,
                    style={
                        'width': '100px',  # Set desired width
                        'border-right': '1px solid #ddd',
                        'display': 'flex',
                        'justify-content': 'center',
                        'align-items': 'center',
                        'flex': '0 0 auto',  # Prevent the column from growing
                    }
                ),
                dbc.Col(
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='results-graph-1'), width=4),
                        dbc.Col(dcc.Graph(id='results-graph-2'), width=4),
                        dbc.Col(dcc.Graph(id='results-graph-3'), width=4),
                    ]),  # Add margin to create space at the bottom of the row
                    style={'flex': '1'},
                ),
            ], className='mb-4'),
            
            # **Second Section: External Dependencies with Vertical Label**
            html.Hr(),
            dbc.Row([
                dbc.Col(
                    html.Div("External Dependencies", style={
                        'writing-mode': 'vertical-rl',
                        'transform': 'rotate(180deg)',
                        'text-align': 'center',
                        'font-size': '28px',
                        'font-weight': 'bold',
                    }),
                    style={
                        'width': '100px',
                        'border-right': '1px solid #ddd',
                        'display': 'flex',
                        'justify-content': 'center',
                        'align-items': 'center',
                        'flex': '0 0 auto',
                    }
                ),
                dbc.Col(
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='context-graph-1'), width=4),
                        dbc.Col(dcc.Graph(id='context-graph-2'), width=4),
                        dbc.Col(dcc.Graph(id='context-graph-3'), width=4),
                    ]),  # Add margin to create space at the bottom of the row
                    style={'flex': '1'},
                ),
            ], className='mb-4'),
        
            # **Third Section: Actions with Vertical Label**
            html.Hr(),
            dbc.Row([
                dbc.Col(
                    html.Div("Actions", style={
                        'writing-mode': 'vertical-rl',
                        'transform': 'rotate(180deg)',
                        'text-align': 'center',
                        'font-size': '28px',
                        'font-weight': 'bold',
                    }),
                    style={
                        'width': '100px',
                        'border-right': '1px solid #ddd',
                        'display': 'flex',
                        'justify-content': 'center',
                        'align-items': 'center',
                        'flex': '0 0 auto',
                    }
                ),
                dbc.Col(
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='actions-graph-1'), width=4),
                        dbc.Col(dcc.Graph(id='actions-graph-2'), width=4),
                        dbc.Col(dcc.Graph(id='actions-graph-3'), width=4),
                    ]),  # Add margin to create space at the bottom of the row
                    style={'flex': '1'},
                ),
            ], className='mb-4'),
        
            dcc.Interval(
                id='interval-component',
                interval=2000,  # Update every 2 seconds
                n_intervals=0
            )
        ], fluid=True)

    def setup_callbacks(self):
        # Callback to update experiment dropdown
        @self.app.callback(
            dash.dependencies.Output('experiment-dropdown', 'options'),
            [dash.dependencies.Input('interval-component', 'n_intervals')]
        )
        def update_experiment_dropdown(n):
            experiments = self.get_available_experiments()
            options = [{'label': experiment, 'value': experiment} for experiment in experiments]
            return options

        # Callback to update episode dropdown based on selected experiment
        @self.app.callback(
            dash.dependencies.Output('episode-dropdown', 'options'),
            dash.dependencies.Output('episode-dropdown', 'value'),
            [dash.dependencies.Input('experiment-dropdown', 'value')]
        )
        def update_episode_dropdown(selected_experiment):
            if selected_experiment is None:
                return [], None
            episodes = self.get_available_episodes(selected_experiment)
            default_episode = episodes[0] if episodes else None
            options = [{'label': ep, 'value': ep} for ep in episodes]
            return options, default_episode

        # Callback to toggle pause/play state and update button label
        @self.app.callback(
            [dash.dependencies.Output('pause-state', 'data'),
            dash.dependencies.Output('pause-button', 'children')],
            [dash.dependencies.Input('pause-button', 'n_clicks')],
            [dash.dependencies.State('pause-state', 'data')]
        )
        def toggle_pause(n_clicks, paused):
            if n_clicks is None:
                raise dash.exceptions.PreventUpdate
            paused = not paused
            label = 'Play' if paused else 'Pause'
            icon = 'fa-play' if paused else 'fa-pause'
            button_content = [html.I(className=f'fas {icon} me-2'), label]
            return paused, button_content
        @self.app.callback(
            dash.dependencies.Output('interval-component', 'interval'),
            [dash.dependencies.Input('interval-input', 'value')],
        )
        def update_interval(interval_value):
            if interval_value is None or interval_value <= 0:
                return dash.no_update  # Prevent updating if the input is invalid
            return interval_value

        # Callback to disable/enable the interval component based on pause state
        @self.app.callback(
            dash.dependencies.Output('interval-component', 'disabled'),
            [dash.dependencies.Input('pause-state', 'data')]
        )
        def update_interval_disabled(paused):
            return paused

        @self.app.callback(
            [dash.dependencies.Output('results-graph-1', 'figure'),
            dash.dependencies.Output('results-graph-2', 'figure'),
            dash.dependencies.Output('results-graph-3', 'figure'),
            dash.dependencies.Output('context-graph-2', 'figure'),
            dash.dependencies.Output('context-graph-1', 'figure'),
            dash.dependencies.Output('context-graph-3', 'figure'),
            dash.dependencies.Output('actions-graph-1', 'figure'),
            dash.dependencies.Output('actions-graph-2', 'figure'),
            dash.dependencies.Output('actions-graph-3', 'figure')],
            [dash.dependencies.Input('experiment-dropdown', 'value'),
            dash.dependencies.Input('episode-dropdown', 'value'),
            dash.dependencies.Input('interval-component', 'n_intervals')]
        )
        def update_graphs(selected_experiment, selected_episode, n_intervals):
            if selected_experiment is None or selected_episode is None:
                return [{}, {}, {}, {}, {}, {}, {}, {}, {}]

            df = self.load_episode_data(selected_experiment, selected_episode)
            if df.empty:
                return [{}, {}, {}, {}, {}, {}, {}, {}, {}]

            # Generate and return figures

            # Compute the current window of data to display (same as before)
            timesteps_per_hour = 4
            timesteps_per_day = timesteps_per_hour * 24  # 96 timesteps per day
            timesteps_to_advance = timesteps_per_hour  # Advance by 4 timesteps every second
            total_timesteps = len(df)
            max_start = total_timesteps - timesteps_per_day
            if max_start <= 0:
                # Not enough data for a full day; display all data
                df_window = df
            else:
                start_timestep = (n_intervals * timesteps_to_advance) % max_start
                end_timestep = start_timestep + timesteps_per_day
                df_window = df.iloc[start_timestep:end_timestep]

            # **Create Datetime Column**
            import datetime
            year = 2023  # Set the appropriate year

            def day_hour_to_datetime(row):
                day_of_year = int(row['day'])
                fractional_hour = row['hour']
                hour = int(fractional_hour)
                minute = int(round((fractional_hour - hour) * 60))

                # Handle potential rounding issues
                if minute == 60:
                    hour += 1
                    minute = 0

                # Handle hour overflow
                if hour == 24:
                    hour = 0
                    day_of_year += 1  # Move to the next day

                # Handle day overflow (year-end)
                if day_of_year > 365:  # Adjust for leap years if necessary
                    day_of_year = 1
                    # Optionally, increment the year if your data spans multiple years
                    # year += 1

                # Create a datetime object
                date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
                date = date.replace(hour=hour, minute=minute)
                return date


            # Apply the function to create a new datetime column
            df_window['datetime'] = df_window.apply(day_hour_to_datetime, axis=1)

            # **First Row: Results**

            # **First Column: Energy Consumption**
            results_fig_1 = go.Figure()
            results_fig_1.add_trace(go.Scatter(
                x=df_window['datetime'],
                y=df_window['bat_total_energy_with_battery_KWh'],
                mode='lines',
                name='Energy Consumption',
                line=dict(color='#F4463B', width=3) 
            ))
            results_fig_1.update_layout(
                title=dict(text='Energy Consumption', font=dict(size=20)),
                yaxis_title=dict(text='Energy (kWh)', font=dict(size=18)),
                xaxis_title=dict(font=dict(size=18)),
                plot_bgcolor='white',
                height=300,
                margin=dict(l=100, r=20, t=40, b=50),  # Increased bottom margin to prevent cropping
                font=dict(size=16),
            )

            # **Second Column: Carbon Emissions**
            results_fig_2 = go.Figure()
            results_fig_2.add_trace(go.Scatter(
                x=df_window['datetime'],
                y=df_window['bat_CO2_footprint']/1000,
                mode='lines',
                name='Carbon Emissions',
                line=dict(color='#7F7F7F', width=3)
            ))
            results_fig_2.update_layout(
                title=dict(text='Carbon Emissions', font=dict(size=20)),
                yaxis_title=dict(text='CO₂ Emissions (kg)', font=dict(size=18)),
                xaxis_title=dict(font=dict(size=18)),
                height=300,
                margin=dict(l=100, r=20, t=40, b=50),  # Increased bottom margin to prevent cropping
                font=dict(size=16),
            )

            # **Third Column: Water Consumption**
            results_fig_3 = go.Figure()
            results_fig_3.add_trace(go.Scatter(
                x=df_window['datetime'],
                y=df_window['dc_water_usage'],
                mode='lines',
                name='Water Consumption',
                line=dict(color='#1F77B4', width=3)
            ))
            results_fig_3.update_layout(
                title=dict(text='Water Consumption', font=dict(size=20)),
                yaxis_title=dict(text='Water Usage (L)', font=dict(size=18)),
                xaxis_title=dict(font=dict(size=18)),
                height=300,
                margin=dict(l=100, r=20, t=40, b=50),  # Increased bottom margin to prevent cropping
                font=dict(size=16),
            )

            # **Second Row: Context**

            # **First Column: Carbon Intensity (bat_avg_CI)**
            context_fig_1 = go.Figure()
            context_fig_1.add_trace(go.Scatter(
                x=df_window['datetime'],
                y=df_window['bat_avg_CI'],
                mode='lines',
                name='Carbon Intensity',
                line=dict(color='#565656', width=3)

            ))
            context_fig_1.update_layout(
                title=dict(text='Power Grid Carbon Intensity', font=dict(size=20)),
                yaxis_title=dict(text='Carbon Intensity (gCO₂/kWh)', font=dict(size=18)),
                height=300,
                margin=dict(l=100, r=20, t=40, b=50),  # Increased bottom margin to prevent cropping
                font=dict(size=16),
            )

            # **Second Column: Original Workload (ls_original_workload)**
            context_fig_2 = go.Figure()
            context_fig_2.add_trace(go.Scatter(
                x=df_window['datetime'],
                y=df_window['ls_original_workload']*100,
                mode='lines',
                name='Original Workload',
                line=dict(color='#F47E39', width=3)
            ))
            context_fig_2.update_layout(
                title=dict(text='Original Workload', font=dict(size=20)),
                yaxis_title=dict(text='Worload (%)', font=dict(size=18)),
                height=300,
                margin=dict(l=100, r=20, t=40, b=50),  # Increased bottom margin to prevent cropping
                font=dict(size=16),
            )
            context_fig_2.update_yaxes(
            range=[0, 100],
        )

            # **Third Column: External Temperature (dc_exterior_ambient_temp)**
            context_fig_3 = go.Figure()
            context_fig_3.add_trace(go.Scatter(
                x=df_window['datetime'],
                y=df_window['dc_exterior_ambient_temp'],
                mode='lines',
                name='External Temperature',
                line=dict(color='#00CC96', width=3)
            ))
            context_fig_3.update_layout(
                title=dict(text='External Temperature', font=dict(size=20)),
                yaxis_title=dict(text='External Temperature (°C)', font=dict(size=18)),
                height=300,
                margin=dict(l=100, r=20, t=40, b=50),  # Increased bottom margin to prevent cropping
                font=dict(size=16),
            )

            # **Third Row: Actions**

            # **First Column: Computed Workload and Tasks in Queue (Separate Subplots)**
            actions_fig_1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2,
                                          subplot_titles=('Computed Workload', 'Tasks in Queue'))

            # Subplot 1: Computed Workload
            actions_fig_1.add_trace(go.Scatter(
                x=df_window['datetime'],
                y=df_window['ls_shifted_workload']*100,
                mode='lines',
                name='Computed Workload',
                line=dict(color='#F4AA36', width=3)
            ), row=1, col=1)
            actions_fig_1.update_yaxes(title_text='Workload (%)', 
                                       range=[0, 110],  # Set y-axis limits
                                       row=1,
                                       col=1)
            

            # Subplot 2: Tasks in Queue (bar chart)
            actions_fig_1.add_trace(go.Bar(
                x=df_window['datetime'],
                y=df_window['ls_tasks_in_queue'],
                name='Tasks in Queue',
                marker_color='#F5D036'
            ), row=2, col=1)

            actions_fig_1.update_yaxes(title_text='Tasks', row=2, col=1)

            # Update layout to style the subplot titles
            actions_fig_1.update_layout(
                annotations=[
                    dict(
                        text='Computed Workload',
                        font=dict(size=20),  # Set font size for subplot title
                        x=0.5,  # Centered on the x-axis
                        xref='paper',
                        y=1.05,  # Position above the first subplot
                        yref='paper',
                        showarrow=False,
                    ),
                    dict(
                        text='Tasks in Queue',
                        font=dict(size=20),  # Set font size for subplot title
                        x=0.5,  # Centered on the x-axis
                        xref='paper',
                        y=0.40,  # Position above the second subplot
                        yref='paper',
                        showarrow=False,
                    )
                ],
                title_font=dict(size=20),
                font=dict(size=16),
                barmode='group',
                bargap=0.15,  # gap between bars of adjacent location coordinates.
                bargroupgap=0.1,  # gap between bars of the same location coordinate.
                showlegend=True,
                legend=dict(
                    x=0.5,         # Horizontal position (0 to 1)
                    y=-0.15,       # Vertical position (set negative to place below the plot)
                    xanchor='center',
                    yanchor='top',
                    orientation='h',  # Display legend horizontally
                ),
                height=400,
                margin=dict(l=100, r=20, t=40, b=50),  # Increased top margin for the subplot titles
            )

            # **Second Column: Cooling Setpoint and Power Consumption (Separate Subplots)**
            actions_fig_2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2,
                                          subplot_titles=('Cooling Setpoint (°C)', 'Energy Consumption (kWh)'))

            # Subplot 1: Cooling Setpoint
            actions_fig_2.add_trace(go.Scatter(
                x=df_window['datetime'],
                y=df_window['dc_crac_setpoint'],
                mode='lines',
                name='Cooling Setpoint',
                line=dict(color='#49B0F5', width=3),
                showlegend=False  # Hide legend for this trace

            ), row=1, col=1)
            actions_fig_2.update_yaxes(title_text='Setpoint (°C)', row=1, col=1)

            # Subplot 2: Power Consumption
            # ITE Power Consumption
            actions_fig_2.add_trace(go.Bar(
                x=df_window['datetime'],
                y=df_window['dc_ITE_total_power_kW']/4,
                name='ITE',
                marker_color='#F4AB48',
                showlegend=True  # Ensure legend is shown (optional, since it's True by default)

            ), row=2, col=1)

            # Cooling System Power Consumption
            actions_fig_2.add_trace(go.Bar(
                x=df_window['datetime'],
                y=df_window['dc_HVAC_total_power_kW']/4,
                name='Cooling System',
                marker_color='#49B3F5',
                showlegend=True  # Ensure legend is shown (optional)
            ), row=2, col=1)

            # Update layout to style the subplot titles
            actions_fig_2.update_layout(
                annotations=[
                    dict(
                        text='Cooling Setpoint (°C)',
                        font=dict(size=20),  # Set font size for the first subplot title
                        x=0.5,  # Centered on the x-axis
                        xref='paper',
                        y=1.05,  # Position above the first subplot
                        yref='paper',
                        showarrow=False,
                    ),
                    dict(
                        text='Energy Consumption (kWh)',
                        font=dict(size=20),  # Set font size for the second subplot title
                        x=0.5,  # Centered on the x-axis
                        xref='paper',
                        y=0.40,  # Position above the second subplot
                        yref='paper',
                        showarrow=False,
                    )
                ],
                title_font=dict(size=20),
                font=dict(size=16),
                barmode='group',
                bargap=0.15,  # gap between bars of adjacent location coordinates.
                bargroupgap=0.1,  # gap between bars of the same location coordinate.
                showlegend=True,
                legend=dict(
                    x=0.5,         # Horizontal position (0 to 1)
                    y=-0.15,       # Vertical position (set negative to place below the plot)
                    xanchor='center',
                    yanchor='top',
                    orientation='h',  # Display legend horizontally
                ),
                height=400,
                margin=dict(l=100, r=20, t=40, b=50),  # Increased top margin for the subplot titles
            )
            actions_fig_2.update_yaxes(title_text='Energy (kWh)', row=2, col=1)

            # **Third Column: Battery SoC and Net Energy (Separate Subplots)**
            actions_fig_3 = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2,
                subplot_titles=('Battery SoC', 'Battery Net Energy')
            )
            # Subplot 1: Battery SoC
            actions_fig_3.add_trace(
                go.Scatter(
                    x=df_window['datetime'],
                    y=df_window['bat_SOC'] * 100,
                    mode='lines',
                    name='Battery SoC (%)',
                    showlegend=False,  # Hide legend for this trace
                    line=dict(color='#9467BD', width=3)
                ),
                row=1, col=1
            )
            
            # Subplot 2: Battery Net Energy
            # Calculate battery_action
            battery_action = df_window['bat_total_energy_with_battery_KWh'] - df_window['bat_total_energy_without_battery_KWh']

            # Create masks for charging and discharging
            charging_mask = battery_action >= 0
            discharging_mask = battery_action <= 0

            # Charging Trace (positive values)
            actions_fig_3.add_trace(
                go.Bar(
                    x=df_window['datetime'][charging_mask],
                    y=battery_action[charging_mask],
                    name='Charging',
                    marker_color='green',
                    showlegend=True
                ),
                row=2, col=1
            )

            # Discharging Trace (negative values)
            actions_fig_3.add_trace(
                go.Bar(
                    x=df_window['datetime'][discharging_mask],
                    y=battery_action[discharging_mask],
                    name='Discharging',
                    marker_color='#F5B93C',
                    showlegend=True
                ),
                row=2, col=1
            )

            # Update layout to style the subplot titles
            actions_fig_3.update_layout(
                annotations=[
                    dict(
                        text='Battery SoC',
                        font=dict(size=20),  # Set font size for the first subplot title
                        x=0.5,  # Centered on the x-axis
                        xref='paper',
                        y=1.05,  # Position above the first subplot
                        yref='paper',
                        showarrow=False,
                    ),
                    dict(
                        text='Battery Net Energy',
                        font=dict(size=20),  # Set font size for the second subplot title
                        x=0.5,  # Centered on the x-axis
                        xref='paper',
                        y=0.4,  # Position above the second subplot
                        yref='paper',
                        showarrow=False,
                    )
                ],
                font=dict(size=16),
                height=400,
                margin=dict(l=100, r=20, t=40, b=50),  # Increased top margin for the subplot titles
                legend=dict(
                    x=0.5,
                    y=-0.15,
                    xanchor='center',
                    yanchor='top',
                    orientation='h',
                ),
            )

            # Add Y-axis labels with updated units
            actions_fig_3.update_yaxes(
                title_text='State of Charge (%)',
                title_font=dict(size=18),
                row=1, col=1
            )
            actions_fig_3.update_yaxes(
                title_text='Energy (kWh)',
                title_font=dict(size=18),
                row=2, col=1
            )


            return [
                results_fig_1, results_fig_2, results_fig_3,
                context_fig_1, context_fig_2, context_fig_3,
                actions_fig_1, actions_fig_2, actions_fig_3
            ]

    def get_available_experiments(self):
        experiments = []
        if os.path.exists(self.evaluation_render_dir):
            for datetime_dir in os.listdir(self.evaluation_render_dir):
                datetime_path = os.path.join(self.evaluation_render_dir, datetime_dir)
                if os.path.isdir(datetime_path):
                    experiments.append(datetime_dir)
        return sorted(experiments)


    def get_available_episodes(self, selected_experiment):
        episodes = []
        datetime_path = os.path.join(self.evaluation_render_dir, selected_experiment)
        if os.path.exists(datetime_path):
            for episode_dir in os.listdir(datetime_path):
                episode_path = os.path.join(datetime_path, episode_dir)
                if os.path.isdir(episode_path):
                    episodes.append(episode_dir)
        # Sort episodes numerically based on the episode number extracted from the directory name
        episodes.sort(key=lambda x: int(x.split('_')[-1]))
        return episodes

    def load_episode_data(self, selected_experiment, selected_episode):
        episode_num = selected_episode.split('_')[-1]
        csv_file = os.path.join(
            self.evaluation_render_dir,
            selected_experiment,
            selected_episode,
            f"all_agents_episode_{episode_num}.csv"
        )
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            return df
        else:
            return pd.DataFrame()

    def run_server(self):
        self.app.run_server(debug=False, use_reloader=False, port=8074)

    def start(self):
        self.server_thread.start()

    def stop(self):
        # Implement a method to stop the Dash server if needed
        pass
