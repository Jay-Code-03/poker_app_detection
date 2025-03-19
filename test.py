import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update, ALL
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import traceback
import time
import json
import re

# Load cytoscape extension
cyto.load_extra_layouts()

def load_decision_tree_data(db_path, stack_min=0, stack_max=25, game_type="heads_up", max_games=1000, exclude_hero=True):
    """Load poker hand data and organize it in a decision tree structure with proper action sequencing"""
    start_time = time.time()
    try:
        conn = sqlite3.connect(db_path)
        
        # Filter for games with specified stack range and type
        game_filter = ""
        if game_type == "heads_up":
            game_filter = "AND g.player_count = 2"
        
        print(f"Starting query for games with stack range {stack_min}-{stack_max}...")
        
        # Use a more efficient approach to get qualified games
        stack_query = f"""
        WITH stack_info AS (
            -- Calculate stack sizes in big blinds for all players
            SELECT 
                gp.game_id,
                gp.player_id,
                gp.initial_stack / g.big_blind AS stack_in_bb,
                g.big_blind
            FROM game_players gp
            JOIN games g ON gp.game_id = g.game_id
            WHERE 1=1 {game_filter}
        ),
        effective_stacks AS (
            -- Calculate minimum stack (effective stack) for each game
            SELECT 
                game_id,
                MIN(stack_in_bb) AS effective_stack_bb
            FROM stack_info
            GROUP BY game_id
        )
        -- Filter by stack range
        SELECT 
            game_id
        FROM effective_stacks
        WHERE effective_stack_bb BETWEEN {stack_min} AND {stack_max}
        LIMIT {max_games}
        """
        
        qualified_games = pd.read_sql_query(stack_query, conn)
        print(f"Found {len(qualified_games)} qualified games. Query took {time.time() - start_time:.2f}s")
        
        if qualified_games.empty:
            conn.close()
            return {"error": f"No games found with effective stacks between {stack_min} and {stack_max} big blinds"}
        
        # Create a temporary table for efficient querying
        print("Creating temporary table...")
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS temp_qualified_games")
        cursor.execute("CREATE TEMPORARY TABLE temp_qualified_games (game_id TEXT)")
        
        # Insert qualified game IDs in batches for better performance
        for i in range(0, len(qualified_games), 1000):
            batch = qualified_games.iloc[i:i+1000]
            cursor.executemany(
                "INSERT INTO temp_qualified_games VALUES (?)", 
                [(game_id,) for game_id in batch['game_id']]
            )
        
        # Get players and positions - INCLUDING is_hero
        print("Retrieving player positions...")
        players_query = """
        SELECT 
            gp.game_id,
            gp.player_id,
            gp.position,
            gp.is_hero,
            gp.initial_stack
        FROM game_players gp
        JOIN temp_qualified_games tqg ON gp.game_id = tqg.game_id
        """
        
        players_df = pd.read_sql_query(players_query, conn)
        
        # Get actions - INCLUDING is_hero by joining with game_players
        print("Retrieving actions...")
        actions_query = """
        SELECT 
            a.game_id,
            a.player_id,
            a.action_round,
            a.simple_action_type,
            a.action_sum,
            a.action_order,
            a.pot_before_action,
            gp.position,
            gp.is_hero  -- Added is_hero flag
        FROM actions a
        JOIN game_players gp ON a.game_id = gp.game_id AND a.player_id = gp.player_id
        JOIN temp_qualified_games tqg ON a.game_id = tqg.game_id
        ORDER BY a.game_id, a.action_order
        """
        
        actions_df = pd.read_sql_query(actions_query, conn)
        conn.close()
        
        print(f"Retrieved {len(actions_df)} actions. Building decision tree...")
        
        # Initialize decision tree with improved structure
        decision_tree = {
            'name': 'root',
            'children': {
                'preflop': {'name': 'preflop', 'children': {}},
                'flop': {'name': 'flop', 'children': {}},
                'turn': {'name': 'turn', 'children': {}},
                'river': {'name': 'river', 'children': {}}
            },
            'exclude_hero': exclude_hero  # Store preference
        }
        
        # Map action rounds to street names
        round_map = {1: 'preflop', 2: 'flop', 3: 'turn', 4: 'river'}
        
        # Process each game's actions
        game_ids = actions_df['game_id'].unique()
        total_games = len(game_ids)
        processed = 0
        
        for game_id in game_ids:
            processed += 1
            if processed % 100 == 0:
                print(f"Processing game {processed}/{total_games}...")
                
            game_actions = actions_df[actions_df['game_id'] == game_id].sort_values('action_order')
            game_players = players_df[players_df['game_id'] == game_id]
            
            # Skip games with no actions
            if game_actions.empty:
                continue
            
            # Process each street in the game
            for street_round, street_actions in game_actions.groupby('action_round'):
                street = round_map.get(street_round, 'unknown')
                
                # Skip unknown streets
                if street not in decision_tree['children']:
                    continue
                
                street_node = decision_tree['children'][street]
                
                # For heads-up games, create a proper action sequence tree
                if game_type == "heads_up":
                    # Skip if not exactly 2 players
                    if len(game_players) != 2:
                        continue
                    
                    # Find BTN/SB and BB positions
                    btn_pos = None
                    bb_pos = None
                    for _, pos_row in game_players.iterrows():
                        if 'BTN' in pos_row['position'] or 'SB' in pos_row['position']:
                            btn_pos = pos_row['position']
                        elif 'BB' in pos_row['position']:
                            bb_pos = pos_row['position']
                    
                    # Skip if positions aren't clear
                    if not btn_pos or not bb_pos:
                        continue
                    
                    # Determine first actor by street
                    # Preflop: BTN acts first, Postflop: BB acts first
                    first_position = btn_pos if street == 'preflop' else bb_pos
                    second_position = bb_pos if first_position == btn_pos else btn_pos
                    
                    # Make sure position nodes exist
                    if first_position not in street_node['children']:
                        street_node['children'][first_position] = {
                            'name': first_position,
                            'children': {},
                            'actions': {},
                            'hero_actions': {}  # Track hero actions separately
                        }
                    
                    # Sort actions by order
                    sorted_actions = street_actions.sort_values('action_order')
                    
                    # Track action sequence and positions
                    current_position = first_position
                    next_position = second_position
                    current_node = street_node['children'][first_position]
                    
                    # Action sequence variables
                    facing_all_in = False
                    is_terminal = False
                    response_required = True  # Flag to indicate if a response is expected
                    
                    # Process each action in sequence
                    for _, action in sorted_actions.iterrows():
                        action_position = action['position']
                        action_type = action['simple_action_type']
                        is_hero = action['is_hero'] == 1  # Check if action is by hero
                        
                        # Skip if this action doesn't match expected position or we're done
                        if action_position != current_position or is_terminal:
                            continue
                        
                        # Handle facing all-in special case
                        if facing_all_in and action_type not in ['call', 'fold', 'all_in_call']:
                            continue
                            
                        # Update action counts in current position node
                        # Track hero actions separately
                        if action_type not in current_node['actions']:
                            current_node['actions'][action_type] = 0
                            current_node['hero_actions'][action_type] = 0
                        
                        # Increment appropriate counter
                        current_node['actions'][action_type] += 1
                        if is_hero:
                            current_node['hero_actions'][action_type] += 1
                        
                        # Create action node if it doesn't exist
                        if action_type not in current_node['children']:
                            current_node['children'][action_type] = {
                                'name': action_type,
                                'children': {},
                                'actions': {},
                                'hero_actions': {}  # Track hero actions separately
                            }
                        
                        # Advance to the action node
                        action_node = current_node['children'][action_type]
                        
                        # Handle terminal actions
                        if action_type == 'fold':
                            # Fold ends the hand immediately
                            is_terminal = True
                            response_required = False
                            continue
                        elif 'check' in action_type:
                            # Check doesn't require response if it's the final action
                            if len(sorted_actions) == 0:  # No more actions
                                is_terminal = True
                                response_required = False
                                continue
                        elif 'all_in' in action_type and 'call' not in action_type:
                            # Player went all-in, opponent faces all-in decision
                            facing_all_in = True
                            
                            # Next player must respond to all-in
                            if next_position not in action_node['children']:
                                action_node['children'][next_position] = {
                                    'name': next_position,
                                    'children': {},
                                    'actions': {},
                                    'hero_actions': {},  # Track hero actions separately
                                    'facing_all_in': True
                                }
                                
                            # Continue with opponent decision
                            current_node = action_node['children'][next_position]
                            current_position, next_position = next_position, current_position
                        elif ('call' in action_type and facing_all_in) or 'all_in_call' in action_type:
                            # All-in call is terminal - showdown
                            is_terminal = True
                            response_required = False
                            continue
                        else:
                            # Regular action - continue sequence with next player
                            if next_position not in action_node['children']:
                                action_node['children'][next_position] = {
                                    'name': next_position,
                                    'children': {},
                                    'actions': {},
                                    'hero_actions': {}  # Track hero actions separately
                                }
                            
                            # Switch to opponent
                            current_node = action_node['children'][next_position]
                            current_position, next_position = next_position, current_position
                else:
                    # Non-HU game implementation with hero action tracking
                    for position, position_actions in street_actions.groupby('position'):
                        if position not in street_node['children']:
                            street_node['children'][position] = {
                                'name': position,
                                'children': {},
                                'actions': {},
                                'hero_actions': {}  # Track hero actions separately
                            }
                        
                        position_node = street_node['children'][position]
                        current_node = position_node
                        
                        # Process each action in sequence
                        for _, action in position_actions.iterrows():
                            action_type = action['simple_action_type']
                            is_hero = action['is_hero'] == 1
                            
                            # Update action counts, tracking hero actions separately
                            if action_type not in current_node['actions']:
                                current_node['actions'][action_type] = 0
                                current_node['hero_actions'][action_type] = 0
                            
                            current_node['actions'][action_type] += 1
                            if is_hero:
                                current_node['hero_actions'][action_type] += 1
                            
                            if action_type not in current_node['children']:
                                current_node['children'][action_type] = {
                                    'name': action_type,
                                    'children': {},
                                    'actions': {},
                                    'hero_actions': {}  # Track hero actions separately
                                }
                            
                            current_node = current_node['children'][action_type]
                            
        # Post-processing: add missing terminal actions
        def complete_terminal_actions(node):
            """Add synthetic terminal actions where needed"""
            if 'facing_all_in' in node and node['facing_all_in']:
                # Player facing all-in must have fold and call options
                if 'call' not in node['children']:
                    node['children']['call'] = {
                        'name': 'call',
                        'children': {},
                        'actions': {},
                        'hero_actions': {},  # Track hero actions separately
                        'is_synthetic': True,
                        'is_terminal': True
                    }
                if 'fold' not in node['children']:
                    node['children']['fold'] = {
                        'name': 'fold',
                        'children': {},
                        'actions': {},
                        'hero_actions': {},  # Track hero actions separately
                        'is_synthetic': True,
                        'is_terminal': True
                    }
                    
            # Process all children recursively
            if 'children' in node:
                for child_name, child_node in list(node['children'].items()):
                    complete_terminal_actions(child_node)
        
        # Apply completion to each street
        for street_name, street_node in decision_tree['children'].items():
            complete_terminal_actions(street_node)
            
        print("Calculating frequencies...")
        # Calculate frequencies and percentages
        calculate_frequencies(decision_tree, exclude_hero)
        
        # Add game count information
        decision_tree['game_count'] = len(qualified_games)
        
        print(f"Decision tree built successfully. Total time: {time.time() - start_time:.2f}s")
        return decision_tree
    
    except Exception as e:
        print(f"Error in load_decision_tree_data: {str(e)}")
        print(traceback.format_exc())
        return {"error": f"Error loading data: {str(e)}"}

def calculate_frequencies(node, exclude_hero=True):
    """Calculate action frequencies and percentages for each node in the decision tree"""
    total_count = 0
    non_hero_count = 0
    
    # Calculate counts for this node
    if 'actions' in node and node['actions']:
        # Sum overall action counts
        total_actions_count = sum(node['actions'].values())
        
        # Calculate non-hero counts by subtracting hero actions
        if 'hero_actions' in node:
            non_hero_actions = {
                action: node['actions'].get(action, 0) - node['hero_actions'].get(action, 0)
                for action in node['actions']
            }
            non_hero_actions_count = sum(max(0, count) for count in non_hero_actions.values())
        else:
            # Fallback if hero_actions not tracked
            non_hero_actions = node['actions']
            non_hero_actions_count = total_actions_count
        
        # Store both sets of counts
        node['total_action_count'] = total_actions_count
        node['non_hero_action_count'] = non_hero_actions_count
        
        # Calculate total percentages
        if total_actions_count > 0:
            node['action_percentages_total'] = {
                action: (count / total_actions_count) * 100 
                for action, count in node['actions'].items()
            }
        
        # Calculate non-hero percentages
        if non_hero_actions_count > 0:
            node['action_percentages_non_hero'] = {
                action: (max(0, non_hero_actions[action]) / non_hero_actions_count) * 100 
                for action in non_hero_actions
                if non_hero_actions[action] > 0  # Only include actions with non-zero counts
            }
        
        # Add to running totals
        total_count += total_actions_count
        non_hero_count += non_hero_actions_count
    
    # Process children recursively
    if 'children' in node:
        for child_name, child_node in node['children'].items():
            # Skip synthetic nodes in count calculation
            if not child_node.get('is_synthetic', False):
                child_total, child_non_hero = calculate_frequencies(child_node, exclude_hero)
                total_count += child_total
                non_hero_count += child_non_hero
            else:
                # For synthetic nodes, just calculate their frequencies
                calculate_frequencies(child_node, exclude_hero)
                
            # Add child count to parent node
            if 'child_counts' not in node:
                node['child_counts'] = {}
                node['child_counts_non_hero'] = {}
            node['child_counts'][child_name] = child_node.get('total_count', 0)
            node['child_counts_non_hero'][child_name] = child_node.get('non_hero_count', 0)
    
    # Store total count in node
    node['total_count'] = total_count
    node['non_hero_count'] = non_hero_count
    
    return total_count, non_hero_count

def create_action_chart(node, exclude_hero=True):
    """Create a chart and table showing action distributions"""
    if 'actions' not in node or not node['actions']:
        return html.Div("No action data available")
    
    # Extract action data
    actions = []
    counts = []
    percentages = []
    
    # Determine which percentages to use
    percentages_key = 'action_percentages_non_hero' if exclude_hero else 'action_percentages_total'
    
    if percentages_key not in node or not node[percentages_key]:
        return html.Div("No action data available for the selected filter")
    
    # Calculate non-hero action counts if needed
    if exclude_hero and 'hero_actions' in node:
        display_counts = {
            action: max(0, node['actions'].get(action, 0) - node['hero_actions'].get(action, 0))
            for action in node['actions']
        }
    else:
        display_counts = node['actions']
        
    # Sort by count (descending)
    for action, count in sorted(display_counts.items(), key=lambda x: x[1], reverse=True):
        # Skip actions with zero count
        if count <= 0:
            continue
        
        actions.append(action)
        counts.append(count)
        percentage = node.get(percentages_key, {}).get(action, 0)
        percentages.append(percentage)
    
    # Create a bar chart
    fig = px.bar(
        x=actions,
        y=counts,
        text=[f"{p:.1f}%" for p in percentages],
        labels={'x': 'Action Type', 'y': 'Count'},
        color=counts,
        color_continuous_scale='Viridis'
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        uniformtext_minsize=10,
        uniformtext_mode='hide',
        height=300,
        margin=dict(t=10, l=50, r=10, b=50)
    )
    
    # Create a table with the same data
    table_header = [
        html.Thead(html.Tr([
            html.Th("Action"),
            html.Th("Count"),
            html.Th("Percentage")
        ]))
    ]
    
    table_rows = []
    for action, count, percentage in zip(actions, counts, percentages):
        table_rows.append(html.Tr([
            html.Td(action),
            html.Td(count),
            html.Td(f"{percentage:.1f}%")
        ]))
    
    table_body = [html.Tbody(table_rows)]
    table = dbc.Table(table_header + table_body, bordered=True, striped=True, hover=True, size="sm")
    
    # Add a note about what's being displayed
    count_type = "opponent" if exclude_hero else "all"
    total_count = node.get('non_hero_action_count', 0) if exclude_hero else node.get('total_action_count', 0)
    note = html.Div([
        html.Span(f"Showing {count_type} actions only. ", className="text-muted"),
        html.Span(f"Total {count_type} actions: {total_count}", className="text-muted")
    ], className="mb-2")
    
    return html.Div([
        note,
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        html.Div(table, style={"maxHeight": "250px", "overflowY": "auto"})
    ])

def create_tree_explorer_app(db_path):
    """Create a game tree explorer with interactive visualization"""
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Define enhanced stylesheet with improved visuals (keep your existing stylesheet)
    default_stylesheet = [
        # Your existing stylesheet here
        {
            'selector': 'node',
            'style': {
                'content': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'width': 'mapData(count, 0, 1000, 40, 100)',
                'height': 'mapData(count, 0, 1000, 40, 100)',
                'background-color': '#7FB3D5',
                'color': '#2C3E50',
                'font-size': '14px',
                'border-width': 2,
                'border-color': '#2471A3',
                'text-wrap': 'wrap',
                'text-max-width': '80px'
            }
        },
        # Position nodes
        {
            'selector': 'node[type = "position"]',
            'style': {
                'background-color': '#85C1E9',
                'border-color': '#3498DB',
                'border-width': 3,
                'shape': 'round-rectangle'
            }
        },
        # Standard action nodes
        {
            'selector': 'node[type = "action"]',
            'style': {
                'background-color': '#F7DC6F',
                'border-color': '#F1C40F',
                'border-width': 2,
                'shape': 'ellipse'
            }
        },
        # Specific action type styling
        {
            'selector': 'node[label *= "fold"]',
            'style': {
                'background-color': '#E74C3C',
                'border-color': '#C0392B'
            }
        },
        {
            'selector': 'node[label *= "raise"]',
            'style': {
                'background-color': '#F39C12',
                'border-color': '#D35400'
            }
        },
        {
            'selector': 'node[label *= "call"]',
            'style': {
                'background-color': '#2ECC71',
                'border-color': '#27AE60'
            }
        },
        {
            'selector': 'node[label *= "check"]',
            'style': {
                'background-color': '#3498DB',
                'border-color': '#2980B9'
            }
        },
        {
            'selector': 'node[label *= "all_in"]',
            'style': {
                'background-color': '#9B59B6',
                'border-color': '#8E44AD'
            }
        },
        # Terminal nodes
        {
            'selector': 'node[terminal = "true"]',
            'style': {
                'shape': 'diamond',
                'width': 'mapData(count, 0, 1000, 35, 80)',
                'height': 'mapData(count, 0, 1000, 35, 80)'
            }
        },
        # Synthetic nodes
        {
            'selector': 'node[synthetic = "true"]',
            'style': {
                'border-style': 'dashed',
                'opacity': 0.7
            }
        },
        # Selected node
        {
            'selector': ':selected',
            'style': {
                'border-color': '#C0392B',
                'border-width': 4,
                'font-weight': 'bold'
            }
        },
        # Edges
        {
            'selector': 'edge',
            'style': {
                'width': 'mapData(count, 0, 1000, 2, 8)',
                'line-color': '#95A5A6',
                'curve-style': 'bezier',
                'target-arrow-shape': 'triangle',
                'target-arrow-color': '#95A5A6',
                'opacity': 'mapData(count, 0, 1000, 0.6, 1)'
            }
        },
        # Synthetic edges
        {
            'selector': 'edge[synthetic = "true"]',
            'style': {
                'line-style': 'dashed',
                'opacity': 0.5,
                'line-color': '#BDC3C7'
            }
        },
        # Different colors for depths
        {
            'selector': '.depth-0',
            'style': {
                'background-color': '#3498DB',
                'font-size': '16px',
                'font-weight': 'bold'
            }
        },
        {
            'selector': '.depth-1',
            'style': {
                'font-size': '15px'
            }
        },
        {
            'selector': '.depth-2',
            'style': {
                'font-size': '14px'
            }
        },
        {
            'selector': '.depth-3',
            'style': {
                'font-size': '13px'
            }
        },
        {
            'selector': '.depth-4',
            'style': {
                'font-size': '12px'
            }
        }
    ]
    
    # Layout for the application
    app.layout = dbc.Container([
        html.H1("Poker Game Tree Explorer", className="my-4 text-center"),
        
        # Game parameters row
        dbc.Row([
            # Stack size filter
            dbc.Col([
                html.Label("Effective Stack Size (BB)"),
                dcc.RangeSlider(
                    id='stack-slider',
                    min=0,
                    max=25,
                    step=1,
                    marks={i: str(i) for i in range(0, 26, 5)},
                    value=[18, 24]  # Default range
                ),
                html.Div(id='stack-slider-output')
            ], width=5),
            
            # Game type filter
            dbc.Col([
                html.Label("Game Type"),
                dcc.Dropdown(
                    id='game-type-dropdown',
                    options=[
                        {'label': 'Heads-Up Only', 'value': 'heads_up'},
                        {'label': 'All Games', 'value': 'all'}
                    ],
                    value='heads_up'
                )
            ], width=3),
            
            # Add hero action toggle switch
            dbc.Col([
                html.Label("Action Frequency"),
                dbc.Switch(
                    id='exclude-hero-switch',
                    label="Exclude Hero Actions",
                    value=True,  # Default to excluding hero
                    className="mt-1"
                )
            ], width=2),
            
            # Apply filters button
            dbc.Col([
                html.Br(),
                dbc.Button(
                    "Apply Filters",
                    id="apply-filters-button",
                    color="primary",
                    className="mt-2"
                )
            ], width=2)
        ], className="mb-4"),
        
        # Loading message container (keep your existing container)
        html.Div(
            id="status-container",
            children=html.Div(
                id="loading-message-container", 
                children=html.H3("Select parameters and click Apply Filters"),
                style={"textAlign": "center", "marginTop": "20px", "marginBottom": "20px"}
            )
        ),
        
        # Main content - tree visualization and node details (keep your existing layout)
        dbc.Row([
            # Tree visualization
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Game Tree Visualization"),
                    dbc.CardBody([
                        # Control buttons for the tree
                        dbc.ButtonGroup([
                            dbc.Button("Reset View", id="reset-view-button", color="secondary", size="sm"),
                            dbc.Button("Back to Previous", id="back-button", color="info", size="sm")
                        ], className="mb-3"),
                        
                        # Enhanced Navigation breadcrumbs with tooltips
                        html.Div([
                            html.Label("Current Path:"),
                            html.Div(id="path-breadcrumbs", className="mb-3")
                        ]),
                        
                        # Street selector for root nodes
                        html.Div([
                            html.Label("Starting Street:"),
                            dcc.RadioItems(
                                id='street-selector',
                                options=[
                                    {'label': 'Preflop', 'value': 'preflop'},
                                    {'label': 'Flop', 'value': 'flop'},
                                    {'label': 'Turn', 'value': 'turn'},
                                    {'label': 'River', 'value': 'river'}
                                ],
                                value='preflop',
                                inline=True,
                                className="mb-3"
                            )
                        ]),
                        
                        # Legend for node types
                        html.Div([
                            html.Label("Legend:"),
                            html.Div([
                                html.Span("Position", className="badge bg-primary me-2"),
                                html.Span("Fold", className="badge bg-danger me-2"),
                                html.Span("Call", className="badge bg-success me-2"),
                                html.Span("Raise", className="badge bg-warning me-2"),
                                html.Span("Check", className="badge bg-info me-2"),
                                html.Span("All-in", className="badge bg-secondary me-2"),
                                html.Span("Terminal", style={"border": "2px solid #E74C3C", "padding": "2px 6px", "margin-right": "10px"}),
                                html.Span("Synthetic", style={"border": "2px dashed #95A5A6", "padding": "2px 6px"})
                            ], className="mb-3")
                        ]),
                        
                        # Network graph visualization
                        html.Div(
                            id="cytoscape-container",
                            children=cyto.Cytoscape(
                                id='cytoscape-tree',
                                layout={
                                    'name': 'dagre', 
                                    'rankDir': 'LR', 
                                    'spacingFactor': 1.8,
                                    'rankSep': 150,
                                    'nodeSep': 100,
                                    'animate': True
                                },
                                style={'width': '100%', 'height': '600px'},
                                elements=[],
                                stylesheet=default_stylesheet,
                                minZoom=0.5,
                                maxZoom=2
                            ),
                            style={"display": "none"}
                        )
                    ])
                ])
            ], width=8),
            
            # Node details panel
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("Node Details", id="node-details-title")),
                    dbc.CardBody([
                        html.Div(id="node-details-content", children=[
                            html.P("Click a node in the tree to see details")
                        ]),
                        html.Hr(),
                        html.Div(id="node-actions-chart")
                    ])
                ]),
                
                # New card for available decisions at current node
                dbc.Card([
                    dbc.CardHeader(html.H4("Available Decisions")),
                    dbc.CardBody([
                        html.Div(id="available-decisions", children=[
                            html.P("Select a node to see available decisions")
                        ])
                    ])
                ], className="mt-3")
            ], width=4)
        ], id="main-content-container", style={"display": "none"}),
        
        # Debug/Status information at the bottom
        dbc.Card([
            dbc.CardHeader("Debug Information"),
            dbc.CardBody(id="debug-info", style={"whiteSpace": "pre-wrap", "fontFamily": "monospace", "fontSize": "12px"})
        ], className="mt-4"),
        
        # Store components to keep track of state
        dcc.Store(id='decision-tree-data'),  # Full tree data
        dcc.Store(id='current-node-path', data=["root"]),  # Current node path
        dcc.Store(id='id-mapping', data={}),  # Mapping from original IDs to simple IDs
        dcc.Store(id='current-node-details'),  # Details of the selected node
        dcc.Store(id='exclude-hero-store', data=True)  # Store hero exclusion preference
    ], fluid=True)
    
    # Callback to update stack slider output
    @app.callback(
        Output('stack-slider-output', 'children'),
        Input('stack-slider', 'value')
    )
    def update_stack_output(value):
        return f"Stack range: {value[0]} BB to {value[1]} BB"
    
    # Store hero exclusion preference
    @app.callback(
        Output('exclude-hero-store', 'data'),
        Input('exclude-hero-switch', 'value')
    )
    def update_hero_exclusion(exclude_hero):
        return exclude_hero
    
    # Callback to load decision tree data
    @app.callback(
        [Output('decision-tree-data', 'data'),
         Output('loading-message-container', 'children'),
         Output('main-content-container', 'style'),
         Output('debug-info', 'children')],
        Input('apply-filters-button', 'n_clicks'),
        [State('stack-slider', 'value'),
         State('game-type-dropdown', 'value'),
         State('exclude-hero-switch', 'value')],  # Get hero exclusion preference
        prevent_initial_call=True
    )
    def load_data(n_clicks, stack_range, game_type, exclude_hero):
        if n_clicks is None:
            return None, html.H3("Select parameters and click Apply Filters"), {"display": "none"}, ""
        
        stack_min, stack_max = stack_range
        debug_info = f"Loading data with stack range {stack_min}-{stack_max}, game type: {game_type}, exclude hero: {exclude_hero}\n"
        
        try:
            # Show loading message
            loading_message = html.Div([
                html.H3("Loading data..."),
                dbc.Spinner(color="primary", type="grow")
            ])
            
# Load decision tree data with hero exclusion preference
            tree_data = load_decision_tree_data(
                db_path, 
                stack_min=stack_min,
                stack_max=stack_max,
                game_type=game_type,
                max_games=2000,
                exclude_hero=exclude_hero
            )
            
            if 'error' in tree_data:
                debug_info += f"Error: {tree_data['error']}"
                return (
                    tree_data, 
                    html.H3(f"Error: {tree_data['error']}"), 
                    {"display": "none"},
                    debug_info
                )
            
            game_count = tree_data.get('game_count', 0)
            success_message = html.H3(f"Data loaded successfully! Analyzed {game_count} games")
            
            debug_info += f"Successfully loaded {game_count} games"
            
            return tree_data, success_message, {"display": "block"}, debug_info
            
        except Exception as e:
            error_trace = traceback.format_exc()
            debug_info += f"Exception: {str(e)}\n\n{error_trace}"
            return {"error": str(e)}, html.H3(f"Error: {str(e)}"), {"display": "none"}, debug_info

    def get_node_by_path(tree_data, path):
        """Get a node in the decision tree by following a path array"""
        if not path:
            return tree_data
            
        current_node = tree_data
        visited_path = []
        
        for i, step in enumerate(path):
            # Root is the starting point (tree_data itself)
            if i == 0 and step == "root":
                visited_path.append("root")
                continue
                
            # Handle composite IDs with hyphens (parent-child format)
            if '-' in step:
                # Extract the actual node name after the last hyphen
                child_name = step.split('-')[-1]
            else:
                child_name = step
            
            visited_path.append(child_name)
                
            # Check if we can navigate to the next step
            if 'children' in current_node and child_name in current_node['children']:
                current_node = current_node['children'][child_name]
            else:
                # For debugging
                print(f"Failed at step {i}: {step} (looking for '{child_name}')")
                print(f"Path so far: {visited_path}")
                if 'children' in current_node:
                    print(f"Available children: {list(current_node['children'].keys())}")
                else:
                    print("No children in current node")
                return None
        
        return current_node

    def get_next_numeric_id(elements):
        """Get the next available numeric ID by examining existing node IDs"""
        # Extract numeric IDs (only those that follow the pattern 'n' + digits)
        numeric_ids = []
        for element in elements:
            if 'data' in element and 'id' in element['data']:
                node_id = element['data']['id']
                # Use regex to match 'n' followed by digits
                match = re.match(r'n(\d+)$', node_id)
                if match:
                    numeric_ids.append(int(match.group(1)))
        
        # If no numeric IDs found, start from 1
        if not numeric_ids:
            return 1
        
        # Return next available ID
        return max(numeric_ids) + 1

    def build_tree_elements(tree_data, start_node_data, start_node_id, max_depth=3, exclude_hero=True):
        """Build tree elements using a simple numeric ID approach to avoid Cytoscape errors"""
        nodes = []
        edges = []
        id_map = {}  # Maps original IDs to simple numeric IDs
        
        # Queue for BFS traversal
        queue = [(start_node_data, start_node_id, 0, None)]
        next_id = 1  # Start from 1 for simplicity
        
        while queue:
            current_data, original_id, depth, parent_simple_id = queue.pop(0)
            
            if depth > max_depth:
                continue
            
            # Create simple numeric ID
            simple_id = f"n{next_id}"
            id_map[original_id] = simple_id
            next_id += 1
            
            # Determine node type
            if any(pos in original_id for pos in ['SB', 'BB', 'BTN']):
                node_type = 'position'
            elif any(action in original_id for action in ['raise', 'bet', 'call', 'fold', 'check', 'all_in']):
                node_type = 'action'
            else:
                node_type = 'street'
            
            # Get display label
            if '-' in original_id:
                label = original_id.split('-')[-1]
            else:
                label = current_data.get('name', original_id)
            
            # Check node flags
            is_synthetic = current_data.get('is_synthetic', False)
            is_terminal = current_data.get('is_terminal', False) or 'fold' in original_id or (
                'call' in original_id and current_data.get('facing_all_in', False)
            )
            
            # Get the appropriate count based on hero exclusion preference
            if exclude_hero:
                node_count = current_data.get('non_hero_count', 0)
            else:
                node_count = current_data.get('total_count', 0)
            
            # Create node
            node = {
                'data': {
                    'id': simple_id,
                    'original_id': original_id,
                    'label': label,
                    'type': node_type,
                    'depth': depth,
                    'count': node_count,
                    'synthetic': is_synthetic,
                    'terminal': is_terminal
                },
                'classes': f"depth-{depth} {node_type}" + 
                         (' synthetic' if is_synthetic else '') +
                         (' terminal' if is_terminal else '')
            }
            
            nodes.append(node)
            
            # Create edge if not root
            if parent_simple_id is not None:
                edge = {
                    'data': {
                        'id': f"e{next_id}",
                        'source': parent_simple_id,
                        'target': simple_id,
                        'count': node_count,
                        'synthetic': is_synthetic
                    },
                    'classes': ('synthetic' if is_synthetic else '')
                }
                edges.append(edge)
                next_id += 1
            
            # Process children if not a terminal node
            if 'children' in current_data and not is_terminal and depth < max_depth:
                # Sort children by appropriate frequency
                if exclude_hero:
                    sorted_children = sorted(
                        current_data['children'].items(),
                        key=lambda x: x[1].get('non_hero_count', 0) if not x[1].get('is_synthetic', False) else -1,
                        reverse=True
                    )
                else:
                    sorted_children = sorted(
                        current_data['children'].items(),
                        key=lambda x: x[1].get('total_count', 0) if not x[1].get('is_synthetic', False) else -1,
                        reverse=True
                    )
                
                for child_name, child_data in sorted_children:
                    child_original_id = f"{original_id}-{child_name}"
                    queue.append((child_data, child_original_id, depth + 1, simple_id))
        
        return nodes, edges, id_map
    
    # Callback to update the tree visualization based on selected street
    @app.callback(
        [Output('cytoscape-container', 'style'),
         Output('cytoscape-tree', 'elements'),
         Output('current-node-path', 'data', allow_duplicate=True),
         Output('id-mapping', 'data')],
        [Input('street-selector', 'value'),
         Input('decision-tree-data', 'data'),
         Input('exclude-hero-store', 'data')],
        prevent_initial_call=True
    )
    def update_tree_visualization(street, tree_data, exclude_hero):
        if tree_data is None or 'error' in tree_data:
            return {"display": "none"}, [], no_update, {}
        
        # Navigate to the selected street
        if street and 'children' in tree_data and street in tree_data['children']:
            street_node = tree_data['children'][street]
            
            # For heads-up poker, look for first position based on street
            if street == 'preflop':
                positions = [pos for pos in street_node.get('children', {}).keys() 
                           if 'SB' in pos or 'BTN' in pos]
            else:
                positions = [pos for pos in street_node.get('children', {}).keys() 
                           if 'BB' in pos]
                
            default_position = positions[0] if positions else None
            
            if default_position:
                # If we have a position, show it as the starting point
                position_node = street_node['children'][default_position]
                
                # Set initial path
                initial_path = ["root", street, default_position]
                
                # Create graph elements starting from position node
                position_orig_id = f"{street}-{default_position}"
                nodes, edges, id_map = build_tree_elements(
                    tree_data, 
                    position_node, 
                    position_orig_id, 
                    max_depth=3,
                    exclude_hero=exclude_hero
                )
                
                # Create and add street node
                street_node_id = "street_" + street
                
                # Get the appropriate count based on hero exclusion preference
                if exclude_hero:
                    node_count = street_node.get('non_hero_count', 0)
                else:
                    node_count = street_node.get('total_count', 0)
                
                street_node_element = {
                    'data': {
                        'id': street_node_id,
                        'original_id': street,
                        'label': street,
                        'type': 'street',
                        'depth': 0,
                        'count': node_count
                    },
                    'classes': 'depth-0 street'
                }
                nodes.append(street_node_element)
                
                # Add mapping for street
                id_map[street] = street_node_id
                
                # Create edge from street to position
                first_position_id = id_map.get(position_orig_id)
                if first_position_id:
                    street_edge = {
                        'data': {
                            'id': f"e_street_pos",
                            'source': street_node_id,
                            'target': first_position_id,
                            'count': position_node.get('non_hero_count' if exclude_hero else 'total_count', 0)
                        }
                    }
                    edges.append(street_edge)
                
                # Combine nodes and edges
                elements = nodes + edges
                
                return {"display": "block"}, elements, initial_path, id_map
            else:
                # Just show the street level
                initial_path = ["root", street]
                
                # Build elements for street node
                nodes, edges, id_map = build_tree_elements(
                    tree_data,
                    street_node,
                    street,
                    max_depth=3,
                    exclude_hero=exclude_hero
                )
                
                elements = nodes + edges
                return {"display": "block"}, elements, initial_path, id_map
            
        return {"display": "none"}, [], no_update, {}
    
    # Callback to update breadcrumbs and show available options
    @app.callback(
        Output('path-breadcrumbs', 'children'),
        [Input('current-node-path', 'data'),
         Input('decision-tree-data', 'data'),
         Input('exclude-hero-store', 'data')]
    )
    def update_breadcrumbs(current_path, tree_data, exclude_hero):
        if not current_path or tree_data is None:
            return html.Div("No path selected")
        
        breadcrumbs = []
        
        # For each step in the path
        for i, step in enumerate(current_path):
            # Add separator except for first item
            if i > 0:
                breadcrumbs.append(html.Span(" > ", className="mx-1"))
            
            # Create button for each step
            btn_class = "btn-primary" if i == len(current_path) - 1 else "btn-outline-primary"
            
            # Extract readable label from path step
            if '-' in step:
                # For composite IDs, show only the last part
                label = step.split('-')[-1]
            else:
                label = step
            
            # Get node at this path
            node_path = current_path[:i+1]
            node = get_node_by_path(tree_data, node_path)
            
            # If node has children, create a dropdown button
            if node and 'children' in node and node['children']:
                # Create dropdown button with available options
                dropdown_items = []
                
                # Sort children by frequency (highest first) based on hero exclusion preference
                if exclude_hero:
                    sorted_children = sorted(
                        node['children'].items(),
                        key=lambda x: x[1].get('non_hero_count', 0) if not x[1].get('is_synthetic', False) else -1,
                        reverse=True
                    )
                else:
                    sorted_children = sorted(
                        node['children'].items(),
                        key=lambda x: x[1].get('total_count', 0) if not x[1].get('is_synthetic', False) else -1,
                        reverse=True
                    )
                
                for child_name, child_data in sorted_children:
                    # Get appropriate count based on hero exclusion
                    if exclude_hero:
                        count = child_data.get('non_hero_count', 0)
                        total = node.get('non_hero_count', 0)
                    else:
                        count = child_data.get('total_count', 0)
                        total = node.get('total_count', 0)
                    
                    percentage = 0
                    if total > 0:
                        percentage = (count / total) * 100
                    
                    # Create item with count and percentage
                    dropdown_items.append(
                        dbc.DropdownMenuItem(
                            f"{child_name} ({count}, {percentage:.1f}%)",
                            id={"type": "path-option", "index": i, "option": child_name},
                            className="path-option-item"
                        )
                    )
                
                breadcrumbs.append(
                    dbc.DropdownMenu(
                        label=label,
                        children=dropdown_items,
                        color=btn_class.replace("btn-", ""),
                        className="mx-1",
                        size="sm"
                    )
                )
            else:
                # Simple button for nodes without children
                breadcrumbs.append(
                    dbc.Button(
                        label,
                        id={"type": "breadcrumb-btn", "index": i},
                        className=f"btn-sm {btn_class} mx-1",
                        size="sm"
                    )
                )
        
        return html.Div(breadcrumbs, className="d-flex align-items-center flex-wrap")
    
    # Callback to navigate using breadcrumbs
    @app.callback(
        [Output('current-node-path', 'data', allow_duplicate=True),
         Output('cytoscape-tree', 'elements', allow_duplicate=True),
         Output('id-mapping', 'data', allow_duplicate=True)],
        [Input({"type": "breadcrumb-btn", "index": ALL}, 'n_clicks'),
         Input({"type": "path-option", "index": ALL, "option": ALL}, 'n_clicks')],
        [State('current-node-path', 'data'),
         State('decision-tree-data', 'data'),
         State('exclude-hero-store', 'data')],
        prevent_initial_call=True
    )
    def navigate_breadcrumb(breadcrumb_clicks, option_clicks, current_path, tree_data, exclude_hero):
        if (not breadcrumb_clicks or not any(n for n in breadcrumb_clicks)) and \
           (not option_clicks or not any(n for n in option_clicks)) or tree_data is None:
            return no_update, no_update, no_update
        
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update
        
        # Get which button was clicked
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        button_data = json.loads(button_id)
        
        if button_data.get('type') == 'breadcrumb-btn':
            # Regular breadcrumb click - navigate to that level
            breadcrumb_index = button_data['index']
            
            # Truncate path to the selected breadcrumb
            new_path = current_path[:breadcrumb_index + 1]
        elif button_data.get('type') == 'path-option':
            # Option selected from dropdown
            path_index = button_data['index']
            option = button_data['option']
            
            # Truncate path to the parent level then add the selected option
            new_path = current_path[:path_index + 1]
            new_path.append(option)
        else:
            return no_update, no_update, no_update
        
        # Get the node at this path
        node = get_node_by_path(tree_data, new_path)
        
        if node is None:
            return no_update, no_update, no_update
        
        # Create an original ID for the node
        if len(new_path) > 1:  # Not root
            original_id = "-".join(new_path[1:])  # Skip "root"
        else:
            original_id = new_path[0]  # Just "root"
        
        # Generate visualization for this node
        nodes, edges, id_map = build_tree_elements(
            tree_data,
            node,
            original_id,
            max_depth=3,
            exclude_hero=exclude_hero
        )
        
        elements = nodes + edges
        
        return new_path, elements, id_map
    
    # Callback to handle node selection in the tree
    @app.callback(
        [Output('current-node-path', 'data', allow_duplicate=True),
         Output('current-node-details', 'data'),
         Output('node-details-title', 'children'),
         Output('node-details-content', 'children'),
         Output('node-actions-chart', 'children'),
         Output('available-decisions', 'children')],
        Input('cytoscape-tree', 'tapNodeData'),
        [State('decision-tree-data', 'data'),
         State('current-node-path', 'data'),
         State('id-mapping', 'data'),
         State('exclude-hero-store', 'data')],
        prevent_initial_call=True
    )
    def handle_node_tap(node_data, tree_data, current_path, id_mapping, exclude_hero):
        if not node_data or tree_data is None or 'error' in tree_data:
            return no_update, None, "Node Details", html.P("No node selected"), html.Div(), html.P("No node selected")
        
        # Get the original ID of the node
        original_id = node_data.get('original_id')
        if not original_id:
            return no_update, None, "Node Details", html.P("Invalid node selection"), html.Div(), html.P("Invalid node selection")
        
        # Build path based on original_id
        if '-' in original_id:
            path_parts = original_id.split('-')
            # If it's a child of a street, add root
            if len(path_parts) == 2 and path_parts[0] in ['preflop', 'flop', 'turn', 'river']:
                new_path = ['root'] + path_parts
            # Otherwise, assume correct structure
            else:
                new_path = ['root'] + path_parts
        else:
            # This is a street or root node
            if original_id in ['preflop', 'flop', 'turn', 'river']:
                new_path = ['root', original_id]
            else:
                new_path = [original_id]
        
        # Get the node data from the tree
        selected_node = get_node_by_path(tree_data, new_path)
        
        if selected_node is None:
            return no_update, None, "Node Details", html.P(f"Node not found in tree for path: {new_path}"), html.Div(), html.P("Node not found")
        
        # Extract node details
        node_name = node_data.get('label', 'Unknown')
        
        # Get appropriate count based on hero exclusion preference
        if exclude_hero:
            node_count = selected_node.get('non_hero_count', 0)
            count_type = "opponent"
        else:
            node_count = selected_node.get('total_count', 0)
            count_type = "total"
            
        node_type = node_data.get('type', 'unknown')
        is_synthetic = node_data.get('synthetic', False)
        is_terminal = node_data.get('terminal', False)
        
        # Create node details display
        details_title = f"Node: {node_name}"
        
        # Node status badges
        status_badges = []
        if is_synthetic:
            status_badges.append(html.Span("SYNTHETIC", className="badge bg-secondary me-2"))
        if is_terminal:
            status_badges.append(html.Span("TERMINAL", className="badge bg-danger me-2"))
        if node_type:
            status_badges.append(html.Span(node_type.upper(), className="badge bg-info me-2"))
            
        # Add hero exclusion badge
        if exclude_hero:
            status_badges.append(html.Span("HERO EXCLUDED", className="badge bg-warning me-2"))
            
        details_content = [
            html.P(status_badges) if status_badges else None,
            html.P(f"{count_type.capitalize()} hands: {node_count}"),
            html.P(f"Path: {' → '.join([p.split('-')[-1] if '-' in p else p for p in new_path])}"),
            html.P(f"Node Type: {node_type}")
        ]
        
        # Special message for synthetic nodes
        if is_synthetic:
            details_content.append(html.Div([
                html.P("This is a synthetic node added to complete the decision tree."),
                html.P("No actual hands in the database follow this exact path.")
            ], className="alert alert-warning"))
            
        # If the node has children, show them
        if 'children' in selected_node and selected_node['children']:
            child_items = []
            
            # Sort children by appropriate frequency
            if exclude_hero:
                sorted_children = sorted(
                    selected_node['children'].items(),
                    key=lambda x: x[1].get('non_hero_count', 0) if not x[1].get('is_synthetic', False) else -1,
                    reverse=True
                )
            else:
                sorted_children = sorted(
                    selected_node['children'].items(),
                    key=lambda x: x[1].get('total_count', 0) if not x[1].get('is_synthetic', False) else -1,
                    reverse=True
                )
                
            for child_name, child_data in sorted_children:
                # Get the appropriate count
                if exclude_hero:
                    child_count = child_data.get('non_hero_count', 0)
                else:
                    child_count = child_data.get('total_count', 0)
                
                child_percentage = (child_count / node_count * 100) if node_count > 0 else 0
                child_synthetic = "SYNTHETIC" if child_data.get('is_synthetic', False) else ""
                
                child_items.append(html.Li([
                    f"{child_name}: {child_count} hands ({child_percentage:.1f}%) {child_synthetic}"
                ]))
            
            details_content.append(html.Div([
                html.H5("Child Nodes:"),
                html.Ul(child_items)
            ]))
        
        # Create action distribution chart with hero exclusion preference
        action_chart = create_action_chart(selected_node, exclude_hero)
        
        # Create available decisions panel
        available_decisions = []
        if 'children' in selected_node and selected_node['children']:
            # Sort children by appropriate frequency
            if exclude_hero:
                sorted_children = sorted(
                    selected_node['children'].items(),
                    key=lambda x: x[1].get('non_hero_count', 0) if not x[1].get('is_synthetic', False) else -1,
                    reverse=True
                )
            else:
                sorted_children = sorted(
                    selected_node['children'].items(),
                    key=lambda x: x[1].get('total_count', 0) if not x[1].get('is_synthetic', False) else -1,
                    reverse=True
                )
                
            for child_name, child_data in sorted_children:
                # Get appropriate count
                if exclude_hero:
                    count = child_data.get('non_hero_count', 0)
                else:
                    count = child_data.get('total_count', 0)
                    
                percentage = (count / node_count * 100) if node_count > 0 else 0
                is_child_synthetic = child_data.get('is_synthetic', False)
                
                # Determine button color based on action type
                btn_color = "secondary"
                if 'fold' in child_name:
                    btn_color = "danger"
                elif 'call' in child_name:
                    btn_color = "success"
                elif 'raise' in child_name or 'bet' in child_name:
                    btn_color = "warning"
                elif 'check' in child_name:
                    btn_color = "info"
                elif 'all_in' in child_name:
                    btn_color = "dark"
                
                # Create button for each option
                option_button = dbc.Button(
                    [
                        f"{child_name}: {percentage:.1f}%",
                        html.Span(f" ({count})", style={"fontSize": "0.8em", "opacity": "0.8"})
                    ],
                    id={"type": "decision-option", "option": child_name},
                    color=btn_color,
                    outline=is_child_synthetic,
                    className="mb-2 me-2",
                    style={"opacity": "0.7" if is_child_synthetic else "1"}
                )
                
                available_decisions.append(option_button)
        
        if not available_decisions:
            available_decisions = [html.P("No further decisions available (terminal node)")]
        
        return new_path, selected_node, details_title, details_content, action_chart, available_decisions
    
    # Callback to handle decision option selection
    @app.callback(
        [Output('current-node-path', 'data', allow_duplicate=True),
         Output('cytoscape-tree', 'elements', allow_duplicate=True),
         Output('id-mapping', 'data', allow_duplicate=True)],
        Input({"type": "decision-option", "option": ALL}, 'n_clicks'),
        [State('current-node-path', 'data'),
         State('decision-tree-data', 'data'),
         State('exclude-hero-store', 'data')],
        prevent_initial_call=True
    )
    def handle_decision_selection(option_clicks, current_path, tree_data, exclude_hero):
        if not option_clicks or not any(n for n in option_clicks) or tree_data is None:
            return no_update, no_update, no_update
        
        ctx = callback_context
        if not ctx.triggered:
            return no_update, no_update, no_update
        
        # Get which option was clicked
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        button_data = json.loads(button_id)
        selected_option = button_data['option']
        
        # Add selected option to path
        new_path = current_path + [selected_option]
        
        # Get the node at this path
        node = get_node_by_path(tree_data, new_path)
        
        if node is None:
            return no_update, no_update, no_update
        
        # Create an original ID for the node
        if len(new_path) > 1:  # Not root
            original_id = "-".join(new_path[1:])  # Skip "root"
        else:
            original_id = new_path[0]  # Just "root"
        
        # Generate visualization for this node
        nodes, edges, id_map = build_tree_elements(
            tree_data,
            node,
            original_id,
            max_depth=3,
            exclude_hero=exclude_hero
        )
        
        elements = nodes + edges
        
        return new_path, elements, id_map
    
    # Callback to expand node on click
    @app.callback(
        [Output('cytoscape-tree', 'elements', allow_duplicate=True),
         Output('id-mapping', 'data', allow_duplicate=True)],
        Input('cytoscape-tree', 'tapNodeData'),
        [State('cytoscape-tree', 'elements'),
         State('decision-tree-data', 'data'),
         State('id-mapping', 'data'),
         State('exclude-hero-store', 'data')],
        prevent_initial_call=True
    )
    def expand_node(node_data, current_elements, tree_data, id_mapping, exclude_hero):
        if not node_data or tree_data is None or 'error' in tree_data:
            return no_update, no_update
        
        # Get original ID and check if already expanded
        original_id = node_data.get('original_id')
        node_id = node_data.get('id')
        is_terminal = node_data.get('terminal', False)
        
        # Don't expand terminal nodes
        if is_terminal:
            return no_update, no_update
            
        # Check if already expanded by looking at existing edges
        outgoing_edges = [e for e in current_elements if 'source' in e.get('data', {}) and e['data']['source'] == node_id]
        if outgoing_edges:
            return no_update, no_update  # Already expanded
            
        # Get the node data from original_id
        if '-' in original_id:
            path_parts = original_id.split('-')
            # If it's a child of a street, add root
            if len(path_parts) == 2 and path_parts[0] in ['preflop', 'flop', 'turn', 'river']:
                node_path = ['root'] + path_parts
            # Otherwise, assume correct structure
            else:
                node_path = ['root'] + path_parts
        else:
            # This is a street or root node
            if original_id in ['preflop', 'flop', 'turn', 'river']:
                node_path = ['root', original_id]
            else:
                node_path = [original_id]
        
        # Get the node from the tree
        selected_node = get_node_by_path(tree_data, node_path)
        
        if selected_node is None or 'children' not in selected_node or not selected_node['children']:
            return no_update, no_update
        
        # Create child nodes and edges
        child_nodes = []
        child_edges = []
        new_id_mapping = id_mapping.copy()
        
        # Get next available ID using the fixed function
        next_id = get_next_numeric_id(current_elements)
        
        # Sort children by frequency
        if exclude_hero:
            sorted_children = sorted(
                selected_node['children'].items(),
                key=lambda x: x[1].get('non_hero_count', 0) if not x[1].get('is_synthetic', False) else -1,
                reverse=True
            )
        else:
            sorted_children = sorted(
                selected_node['children'].items(),
                key=lambda x: x[1].get('total_count', 0) if not x[1].get('is_synthetic', False) else -1,
                reverse=True
            )
            
        for child_name, child_data in sorted_children:
            # Create child ID
            child_original_id = f"{original_id}-{child_name}"
            child_simple_id = f"n{next_id}"
            next_id += 1
            
            # Store ID mapping
            new_id_mapping[child_original_id] = child_simple_id
            
            # Determine node type
            if any(pos in child_name for pos in ['SB', 'BB', 'BTN']):
                child_type = 'position'
            elif any(action in child_name for action in ['raise', 'bet', 'call', 'fold', 'check', 'all_in']):
                child_type = 'action'
            else:
                child_type = 'street'
            
            # Check special flags
            is_synthetic = child_data.get('is_synthetic', False)
            is_terminal = child_data.get('is_terminal', False) or 'fold' in child_name or (
                'call' in child_name and child_data.get('facing_all_in', False)
            )
            
            # Get the appropriate count
            if exclude_hero:
                node_count = child_data.get('non_hero_count', 0)
            else:
                node_count = child_data.get('total_count', 0)
            
            # Create child node
            child_node = {
                'data': {
                    'id': child_simple_id,
                    'original_id': child_original_id,
                    'label': child_name,
                    'type': child_type,
                    'count': node_count,
                    'synthetic': is_synthetic,
                    'terminal': is_terminal
                },
                'classes': f"{child_type}" + 
                         (' synthetic' if is_synthetic else '') +
                         (' terminal' if is_terminal else '')
            }
            
            child_nodes.append(child_node)
            
            # Create edge
            edge_id = f"e{next_id}"
            next_id += 1
            
            child_edge = {
                'data': {
                    'id': edge_id,
                    'source': node_id,
                    'target': child_simple_id,
                    'count': node_count,
                    'synthetic': is_synthetic
                },
                'classes': ('synthetic' if is_synthetic else '')
            }
            
            child_edges.append(child_edge)
        
        # Add new elements to existing ones
        updated_elements = current_elements + child_nodes + child_edges
        
        return updated_elements, new_id_mapping
    
    # Callback to handle hero exclusion toggle during analysis
    @app.callback(
        [Output('cytoscape-tree', 'elements', allow_duplicate=True),
         Output('node-actions-chart', 'children', allow_duplicate=True)],
        Input('exclude-hero-switch', 'value'),
        [State('decision-tree-data', 'data'),
         State('current-node-path', 'data'),
         State('street-selector', 'value')],
        prevent_initial_call=True
    )
    def toggle_hero_exclusion(exclude_hero, tree_data, current_path, street):
        if tree_data is None or 'error' in tree_data:
            return no_update, no_update
        
        # Get the current selected node
        selected_node = get_node_by_path(tree_data, current_path)
        
        # Update the action chart with the new hero exclusion setting
        action_chart = create_action_chart(selected_node, exclude_hero) if selected_node else html.Div()
        
        # Reload the tree visualization with the new setting
        # This will trigger update_tree_visualization callback which will respect the exclude_hero setting
        
        return no_update, action_chart
    
    # Callback for the reset view button
    @app.callback(
        [Output('current-node-path', 'data', allow_duplicate=True),
         Output('cytoscape-tree', 'elements', allow_duplicate=True),
         Output('id-mapping', 'data', allow_duplicate=True)],
        Input('reset-view-button', 'n_clicks'),
        [State('street-selector', 'value'),
         State('decision-tree-data', 'data'),
         State('exclude-hero-store', 'data')],
        prevent_initial_call=True
    )
    def reset_view(n_clicks, street, tree_data, exclude_hero):
        if n_clicks is None or tree_data is None:
            return no_update, no_update, no_update
        
        # Reset to the street level
        if 'children' in tree_data and street in tree_data['children']:
            street_node = tree_data['children'][street]
            
            # Look for appropriate position to show first based on street
            if street == 'preflop':
                positions = [pos for pos in street_node.get('children', {}).keys() 
                          if 'SB' in pos or 'BTN' in pos]
            else:
                positions = [pos for pos in street_node.get('children', {}).keys() 
                          if 'BB' in pos]
                
            default_position = positions[0] if positions else None
            
            if default_position:
                # If we have a position, show it as the starting point
                position_node = street_node['children'][default_position]
                
                # Set initial path
                initial_path = ["root", street, default_position]
                
                # Create elements
                position_orig_id = f"{street}-{default_position}"
                nodes, edges, id_map = build_tree_elements(
                    tree_data,
                    position_node,
                    position_orig_id,
                    max_depth=3,
                    exclude_hero=exclude_hero
                )
                
                # Add street node
                street_node_id = "street_" + street
                
                # Get appropriate count
                if exclude_hero:
                    node_count = street_node.get('non_hero_count', 0)
                else:
                    node_count = street_node.get('total_count', 0)
                    
                street_node_element = {
                    'data': {
                        'id': street_node_id,
                        'original_id': street,
                        'label': street,
                        'type': 'street',
                        'depth': 0,
                        'count': node_count
                    },
                    'classes': 'depth-0 street'
                }
                nodes.append(street_node_element)
                
                # Add mapping for street
                id_map[street] = street_node_id
                
                # Add edge from street to position
                first_position_id = id_map.get(position_orig_id)
                if first_position_id:
                    street_edge = {
                        'data': {
                            'id': f"e_street_pos",
                            'source': street_node_id,
                            'target': first_position_id,
                            'count': position_node.get('non_hero_count' if exclude_hero else 'total_count', 0)
                        }
                    }
                    edges.append(street_edge)
                
                elements = nodes + edges
                return initial_path, elements, id_map
            else:
                # Just show the street level
                initial_path = ["root", street]
                nodes, edges, id_map = build_tree_elements(
                    tree_data,
                    street_node,
                    street,
                    max_depth=3,
                    exclude_hero=exclude_hero
                )
                
                elements = nodes + edges
                return initial_path, elements, id_map
                
        return no_update, no_update, no_update
    
    # Callback for the back button
    @app.callback(
        [Output('current-node-path', 'data', allow_duplicate=True),
         Output('cytoscape-tree', 'elements', allow_duplicate=True),
         Output('id-mapping', 'data', allow_duplicate=True)],
        Input('back-button', 'n_clicks'),
        [State('current-node-path', 'data'),
         State('decision-tree-data', 'data'),
         State('exclude-hero-store', 'data')],
        prevent_initial_call=True
    )
    def go_back(n_clicks, current_path, tree_data, exclude_hero):
        if n_clicks is None or tree_data is None or len(current_path) <= 2:
            return no_update, no_update, no_update
        
        # Go back one level
        new_path = current_path[:-1]
        
        # Get the node at the new path
        parent_node = get_node_by_path(tree_data, new_path)
        
        if parent_node is None:
            return no_update, no_update, no_update
        
        # Create an original ID for the parent
        if len(new_path) > 1:  # Not root
            parent_id = "-".join(new_path[1:])  # Skip "root"
        else:
            parent_id = new_path[0]  # Just use "root"
            
        # Generate elements for parent node
        nodes, edges, id_map = build_tree_elements(
            tree_data,
            parent_node,
            parent_id,
            max_depth=3,
            exclude_hero=exclude_hero
        )
        
        elements = nodes + edges
        
        return new_path, elements, id_map
    
    return app

def main():
    import webbrowser
    
    db_path = "poker_analysis_optimized.db"
    app = create_tree_explorer_app(db_path)
    
    # Print a clearer message
    print("\n")
    print("=" * 60)
    print("Poker Game Tree Explorer is running!")
    print("Open your web browser and go to: http://127.0.0.1:8050/")
    print("=" * 60)
    print("\n")
    
    # Automatically open the browser after a short delay
    webbrowser.open('http://127.0.0.1:8050/', new=1, autoraise=True)
    
    # Run the app
    app.run_server(debug=True)

if __name__ == "__main__":
    main()