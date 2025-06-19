import os
import numpy as np
import pandas as pd
import datetime
import time
import random

from matplotlib.patches import Circle

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
plt.rcParams['figure.figsize'] = (12, 10)

def generate_customer_profiles_table(n_customers, random_state=0):
    """
    Generate customer profiles with geographical and spending properties
    """
    np.random.seed(random_state)
    customer_id_properties = []
    
    # Generate customer properties from random distributions 
    for customer_id in range(n_customers):
        x_customer_id = np.random.uniform(0, 500)
        y_customer_id = np.random.uniform(0, 500)
        mean_amount = np.random.uniform(5, 100)  # Arbitrary (but sensible) value
        std_amount = mean_amount / 2  # Arbitrary (but sensible) value  
        mean_nb_tx_per_day = np.random.uniform(0, 4)  # Arbitrary (but sensible) value
        
        customer_id_properties.append([customer_id,
                                     x_customer_id, y_customer_id,
                                     mean_amount, std_amount,
                                     mean_nb_tx_per_day])
    
    customer_profiles_table = pd.DataFrame(customer_id_properties, 
                                         columns=['CUSTOMER_ID',
                                                'x_customer_id', 'y_customer_id',
                                                'mean_amount', 'std_amount',
                                                'mean_nb_tx_per_day'])
    return customer_profiles_table


def generate_terminal_profiles_table(n_terminals, random_state=1):
    """
    Generate terminal profiles with geographical location and MCC codes
    """
    np.random.seed(random_state)
    terminal_id_properties = []
    
    # Common MCC codes with their descriptions and probability weights
    mcc_codes = {
        5411: 'Grocery Stores',        # High frequency
        5541: 'Service Stations',      # High frequency  
        5812: 'Eating Places',         # High frequency
        5311: 'Department Stores',     # Medium frequency
        5944: 'Jewelry Stores',        # Low frequency, high value
        5912: 'Drug Stores',           # Medium frequency
        5999: 'Miscellaneous Retail', # Medium frequency
        4121: 'Taxicabs',             # Medium frequency
        5734: 'Computer Software',     # Low frequency, high value
        5732: 'Electronics Stores',    # Medium frequency
        5691: 'Mens Clothing',        # Low frequency
        5651: 'Family Clothing',      # Medium frequency
        4411: 'Cruise Lines',         # Very low frequency, very high value
        3000: 'Airlines',             # Low frequency, high value
        4112: 'Passenger Railways',    # Low frequency
        5921: 'Package Stores',       # Medium frequency
        5993: 'Cigar Stores',         # Low frequency
        7011: 'Hotels',               # Low frequency, high value
        5661: 'Shoe Stores',          # Low frequency
        5200: 'Home Supply Stores'    # Medium frequency
    }
    
    mcc_list = list(mcc_codes.keys())
    # Weights for MCC selection (higher weight = more common)
    mcc_weights = [0.15, 0.12, 0.18, 0.08, 0.02, 0.09, 0.07, 0.05, 0.03, 0.06, 
                   0.03, 0.05, 0.01, 0.02, 0.02, 0.04, 0.01, 0.02, 0.03, 0.04]
    # Normalize weights to sum to 1
    mcc_weights = np.array(mcc_weights)
    mcc_weights = mcc_weights / mcc_weights.sum()
    
    # Generate terminal properties from random distributions
    for terminal_id in range(n_terminals):
        x_terminal_id = np.random.uniform(0, 500)
        y_terminal_id = np.random.uniform(0, 500)
        
        # Assign MCC code based on weighted probabilities
        mcc_code = np.random.choice(mcc_list, p=mcc_weights)
        
        terminal_id_properties.append([terminal_id,
                                     x_terminal_id, y_terminal_id,
                                     mcc_code])
    
    terminal_profiles_table = pd.DataFrame(terminal_id_properties, 
                                         columns=['TERMINAL_ID',
                                                'x_terminal_id', 'y_terminal_id',
                                                'MCC_CODE'])
    return terminal_profiles_table

def get_list_terminals_within_radius(customer_profile, x_y_terminals, r):
    """
    Find terminals within radius r of customer location
    """
    # Use numpy arrays to speed up computations
    # Location (x,y) of customer as numpy array
    x_y_customer = customer_profile[['x_customer_id', 'y_customer_id']].values.astype(float)
    
    # Squared difference in coordinates between customer and terminal locations
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)
    
    # Sum along rows and compute squared root to get distance
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))
    
    # Get the indices of terminals which are at a distance less than r
    available_terminals = list(np.where(dist_x_y < r)[0])
    
    # Return the list of terminal IDs
    return available_terminals

def get_transaction_type_probabilities(mcc_code):
    """
    Get transaction type probabilities based on MCC code
    Returns probabilities for [swipe, chip, card-on-file, ecom, tap]
    """
    # Default probabilities
    default_probs = [0.25, 0.35, 0.15, 0.15, 0.10]
    
    # MCC-specific probability adjustments
    mcc_adjustments = {
        5411: [0.20, 0.40, 0.10, 0.05, 0.25],  # Grocery - more tap/chip
        5541: [0.30, 0.30, 0.15, 0.05, 0.20],  # Gas stations - more swipe
        5812: [0.15, 0.30, 0.10, 0.25, 0.20],  # Restaurants - more ecom (delivery)
        5944: [0.10, 0.40, 0.20, 0.25, 0.05],  # Jewelry - more chip/ecom
        5734: [0.05, 0.15, 0.25, 0.50, 0.05],  # Software - mostly ecom
        4411: [0.05, 0.20, 0.30, 0.40, 0.05],  # Cruise - more card-on-file/ecom
        3000: [0.05, 0.25, 0.35, 0.30, 0.05],  # Airlines - more card-on-file/ecom
        7011: [0.10, 0.30, 0.40, 0.15, 0.05],  # Hotels - more card-on-file
    }
    
    return mcc_adjustments.get(mcc_code, default_probs)

def generate_transactions_table(customer_profile, terminal_profiles_table, start_date, nb_days, r):
    """
    Generate transactions table with MCC codes and transaction types
    """
    customer_transactions = []
    random.seed(int(customer_profile.CUSTOMER_ID))
    np.random.seed(int(customer_profile.CUSTOMER_ID))
    
    # Transaction types
    tx_types = ['swipe', 'chip', 'card-on-file', 'ecom', 'tap']
    
    # For all days
    for day in range(nb_days):
        # Random number of transactions for that day
        nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)
        
        # If nb_tx positive, let us generate transactions
        if nb_tx > 0:
            for tx in range(nb_tx):
                # Time of transaction: Around noon, std 20000 seconds
                time_tx = int(np.random.normal(86400/2, 20000))
                
                # If transaction time between 0 and 86400, let us keep it
                if (time_tx > 0) and (time_tx < 86400):
                    # Amount is drawn from a normal distribution
                    amount = np.random.normal(customer_profile.mean_amount, customer_profile.std_amount)
                    
                    # If amount negative, draw from a uniform distribution
                    if amount < 0:
                        amount = np.random.uniform(0, customer_profile.mean_amount * 2)
                    amount = np.round(amount, decimals=2)
                    
                    if len(customer_profile.available_terminals) > 0:
                        # Select terminal
                        terminal_id = random.choice(customer_profile.available_terminals)
                        
                        # Get MCC code for this terminal
                        mcc_code = terminal_profiles_table[terminal_profiles_table.TERMINAL_ID == terminal_id]['MCC_CODE'].iloc[0]
                        
                        # Get transaction type probabilities based on MCC
                        tx_type_probs = get_transaction_type_probabilities(mcc_code)
                        
                        # Select transaction type based on probabilities
                        tx_type = np.random.choice(tx_types, p=tx_type_probs)
                        
                        # Calculate distance between customer and terminal
                        x_customer = customer_profile['x_customer_id']
                        y_customer = customer_profile['y_customer_id']
                        terminal_row = terminal_profiles_table[terminal_profiles_table.TERMINAL_ID == terminal_id].iloc[0]
                        x_terminal = terminal_row['x_terminal_id']
                        y_terminal = terminal_row['y_terminal_id']
                        dist = np.sqrt((x_customer - x_terminal)**2 + (y_customer - y_terminal)**2)
                        # Use r from argument
                        r_val = r 
                        if dist <= 0.90 * r_val:
                            dom_xb = 'domestic'
                        else:
                            dom_xb = 'xb'
                        
                        customer_transactions.append([time_tx + day * 86400, day,
                                                    customer_profile.CUSTOMER_ID,
                                                    terminal_id, amount, mcc_code, tx_type, dom_xb])
    
    customer_transactions = pd.DataFrame(customer_transactions, 
                                       columns=['TX_TIME_SECONDS', 'TX_TIME_DAYS', 
                                               'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT',
                                               'MCC_CODE', 'TX_TYPE', 'TX_DOM_XB'])
    
    if len(customer_transactions) > 0:
        customer_transactions['TX_DATETIME'] = pd.to_datetime(customer_transactions["TX_TIME_SECONDS"], 
                                                            unit='s', origin=start_date)
        customer_transactions = customer_transactions[['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 
                                                     'TX_AMOUNT', 'MCC_CODE', 'TX_TYPE', 'TX_DOM_XB',
                                                     'TX_TIME_SECONDS', 'TX_TIME_DAYS']]
    
    return customer_transactions


def generate_dataset(n_customers, n_terminals, nb_days, start_date, r):
    """
    Generate complete dataset with customer profiles, terminal profiles, and transactions
    """
    start_time = time.time()
    customer_profiles_table = generate_customer_profiles_table(n_customers, random_state=0)
    print(f"Time to generate customer profiles table: {time.time()-start_time:.2}s")
    
    start_time = time.time()
    terminal_profiles_table = generate_terminal_profiles_table(n_terminals, random_state=1)
    print(f"Time to generate terminal profiles table: {time.time()-start_time:.2}s")
    
    start_time = time.time()
    x_y_terminals = terminal_profiles_table[['x_terminal_id', 'y_terminal_id']].values.astype(float)
    customer_profiles_table['available_terminals'] = customer_profiles_table.apply(
        lambda x: get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    customer_profiles_table['nb_terminals'] = customer_profiles_table.available_terminals.apply(len)
    print(f"Time to associate terminals to customers: {time.time()-start_time:.2}s")
    
    start_time = time.time()
    transactions_df = customer_profiles_table.groupby('CUSTOMER_ID').apply(
        lambda x: generate_transactions_table(x.iloc[0], terminal_profiles_table, start_date=start_date, nb_days=nb_days, r=r)
    ).reset_index(drop=True)
    print(f"Time to generate transactions: {time.time()-start_time:.2}s")
    
    # Sort transactions chronologically
    transactions_df = transactions_df.sort_values('TX_DATETIME')
    
    # Reset indices, starting from 0
    transactions_df.reset_index(inplace=True, drop=True)
    transactions_df.reset_index(inplace=True)
    
    # TRANSACTION_ID are the dataframe indices, starting from 0
    transactions_df.rename(columns={'index': 'TRANSACTION_ID'}, inplace=True)
    
    return (customer_profiles_table, terminal_profiles_table, transactions_df)

def visualize_customer_terminal_locations(customer_profiles_table, terminal_profiles_table):
    """
    Visualize the locations of all customers and terminals on a scatter plot.
    Customers: blue dots, Terminals: red x's
    """
    plt.figure(figsize=(10, 8))
    # Plot customers
    plt.scatter(customer_profiles_table['x_customer_id'], customer_profiles_table['y_customer_id'],
                c='blue', label='Customers', alpha=0.5, s=20)
    # Plot terminals
    plt.scatter(terminal_profiles_table['x_terminal_id'], terminal_profiles_table['y_terminal_id'],
                c='red', marker='s', label='Terminals', alpha=0.7, s=40)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Customer and Terminal Locations')
    plt.legend()
    plt.tight_layout()
    plt.show()   


def add_frauds(customer_profiles_table: pd.DataFrame, 
               terminal_profiles_table: pd.DataFrame, 
               transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sophisticated fraud scenarios with realistic patterns
    
    Args:
        customer_profiles_table: Customer data
        terminal_profiles_table: Terminal data  
        transactions_df: Transaction data
        
    Returns:
        DataFrame with fraud labels added
    """
    if len(transactions_df) == 0:
        print("No transactions to add fraud to")
        return transactions_df
        
    print("Adding fraud scenarios...")
    
    # Initialize fraud columns
    transactions_df['TX_FRAUD'] = 0
    transactions_df['TX_FRAUD_SCENARIO'] = 0
    
    # Scenario 1: High-value transactions (MCC-aware)
    high_value_mccs = {5944: 500, 5734: 300, 4411: 800, 3000: 400, 7011: 350}
    
    fraud_count_s1 = 0
    for mcc, threshold in high_value_mccs.items():
        mask = (transactions_df.MCC_CODE == mcc) & (transactions_df.TX_AMOUNT > threshold)
        transactions_df.loc[mask, 'TX_FRAUD'] = 1
        transactions_df.loc[mask, 'TX_FRAUD_SCENARIO'] = 1
        fraud_count_s1 += mask.sum()
    
    # Other MCCs with general threshold
    other_mask = (~transactions_df.MCC_CODE.isin(high_value_mccs.keys())) & (transactions_df.TX_AMOUNT > 220)
    transactions_df.loc[other_mask, 'TX_FRAUD'] = 1
    transactions_df.loc[other_mask, 'TX_FRAUD_SCENARIO'] = 1
    fraud_count_s1 += other_mask.sum()
    
    print(f"✓ Scenario 1 (High-value): {fraud_count_s1} frauds")
    
    # Scenario 2: Compromised terminals
    fraud_count_s2 = 0
    max_days = transactions_df.TX_TIME_DAYS.max()
    
    for day in range(0, max_days, 14):  # Every 2 weeks
        n_compromised = min(3, len(terminal_profiles_table))
        compromised_terminals = terminal_profiles_table.TERMINAL_ID.sample(
            n=n_compromised, random_state=day
        ).values
        
        # Transactions at compromised terminals
        mask = (
            (transactions_df.TX_TIME_DAYS >= day) &
            (transactions_df.TX_TIME_DAYS < day + 28) &
            (transactions_df.TERMINAL_ID.isin(compromised_terminals))
        )
        
        compromised_tx = transactions_df[mask]
        
        # Transaction type fraud probabilities
        fraud_probs = {'swipe': 0.6, 'chip': 0.4, 'card-on-file': 0.8, 'ecom': 0.5, 'tap': 0.3}
        
        for idx in compromised_tx.index:
            tx_type = transactions_df.loc[idx, 'TX_TYPE']
            if np.random.random() < fraud_probs.get(tx_type, 0.5):
                transactions_df.loc[idx, 'TX_FRAUD'] = 1
                transactions_df.loc[idx, 'TX_FRAUD_SCENARIO'] = 2
                fraud_count_s2 += 1
    
    print(f"✓ Scenario 2 (Compromised terminals): {fraud_count_s2} frauds")
    
    # Scenario 3: Compromised cards
    fraud_count_s3 = 0
    
    for day in range(0, max_days, 21):  # Every 3 weeks
        n_compromised = min(5, len(customer_profiles_table))
        compromised_customers = customer_profiles_table.CUSTOMER_ID.sample(
            n=n_compromised, random_state=day
        ).values
        
        # Transactions by compromised customers
        mask = (
            (transactions_df.TX_TIME_DAYS >= day) &
            (transactions_df.TX_TIME_DAYS < day + 14) &
            (transactions_df.CUSTOMER_ID.isin(compromised_customers))
        )
        
        compromised_tx = transactions_df[mask]
        
        if len(compromised_tx) > 0:
            # Prefer CNP transactions for card compromise
            cnp_tx = compromised_tx[compromised_tx.TX_TYPE.isin(['ecom', 'card-on-file'])]
            other_tx = compromised_tx[~compromised_tx.TX_TYPE.isin(['ecom', 'card-on-file'])]
            
            # Select fraudulent transactions
            fraud_indices = []
            target_frauds = max(1, len(compromised_tx) // 4)
            
            # Prefer CNP transactions (70% of frauds)
            if len(cnp_tx) > 0:
                n_cnp_frauds = min(len(cnp_tx), int(target_frauds * 0.7))
                fraud_indices.extend(
                    np.random.choice(cnp_tx.index, n_cnp_frauds, replace=False)
                )
            
            # Fill remaining with other transaction types
            remaining = target_frauds - len(fraud_indices)
            if remaining > 0 and len(other_tx) > 0:
                n_other_frauds = min(len(other_tx), remaining)
                fraud_indices.extend(
                    np.random.choice(other_tx.index, n_other_frauds, replace=False)
                )
            
            # Apply fraud indicators
            for idx in fraud_indices:
                transactions_df.loc[idx, 'TX_AMOUNT'] *= np.random.uniform(2, 6)  # Increase amount
                transactions_df.loc[idx, 'TX_FRAUD'] = 1
                transactions_df.loc[idx, 'TX_FRAUD_SCENARIO'] = 3
                fraud_count_s3 += 1
    
    print(f"✓ Scenario 3 (Compromised cards): {fraud_count_s3} frauds")
    
    total_frauds = transactions_df.TX_FRAUD.sum()
    fraud_rate = total_frauds / len(transactions_df) * 100
    print(f"✓ Total frauds: {total_frauds} ({fraud_rate:.2f}% of transactions)")
    
    return transactions_df


def analyze_dataset(transactions_df):
    """
    Analyze the generated dataset and print summary statistics
    """
    print(f"\nDataset Summary:")
    print(f"Total transactions: {len(transactions_df):,}")
    print(f"Date range: {transactions_df.TX_DATETIME.min()} to {transactions_df.TX_DATETIME.max()}")
    print(f"Unique customers: {transactions_df.CUSTOMER_ID.nunique():,}")
    print(f"Unique terminals: {transactions_df.TERMINAL_ID.nunique():,}")
    print(f"Unique MCC codes: {transactions_df.MCC_CODE.nunique()}")
    
    print(f"\nTransaction Type Distribution:")
    tx_type_dist = transactions_df.TX_TYPE.value_counts(normalize=True) * 100
    for tx_type, pct in tx_type_dist.items():
        print(f"  {tx_type}: {pct:.1f}%")

    print(f"\nTransaction Geography Distribution:")
    tx_geo_dist = transactions_df.TX_DOM_XB.value_counts(normalize=True) * 100
    for tx_type, pct in tx_geo_dist.items():
        print(f"  {tx_type}: {pct:.1f}%")
    
    print(f"\nTop 10 MCC Codes:")
    mcc_dist = transactions_df.MCC_CODE.value_counts().head(10)
    for mcc, count in mcc_dist.items():
        print(f"  MCC {mcc}: {count:,} transactions")
    
    if 'TX_FRAUD' in transactions_df.columns:
        fraud_rate = transactions_df.TX_FRAUD.mean() * 100
        print(f"\nFraud Statistics:")
        print(f"  Total fraudulent transactions: {transactions_df.TX_FRAUD.sum():,}")
        print(f"  Fraud rate: {fraud_rate:.3f}%")
        
        if 'TX_FRAUD_SCENARIO' in transactions_df.columns:
            fraud_scenarios = transactions_df[transactions_df.TX_FRAUD == 1].TX_FRAUD_SCENARIO.value_counts()
            for scenario, count in fraud_scenarios.items():
                print(f"  Scenario {scenario}: {count:,} frauds")


def visualize_customer_terminals(customer_profiles_table, terminal_profiles_table, 
                                customer_id, r=100, show_mcc_info=True):
    """
    Visualize a single customer and their accessible terminals with domestic/xb classification
    
    Parameters:
    - customer_profiles_table: DataFrame with customer profiles
    - terminal_profiles_table: DataFrame with terminal profiles  
    - customer_id: ID of customer to visualize
    - r: radius for terminal accessibility
    - show_mcc_info: whether to show MCC code information
    """
    
    # Get customer data
    customer_data = customer_profiles_table[customer_profiles_table['CUSTOMER_ID'] == customer_id].iloc[0]
    customer_x = customer_data['x_customer_id']
    customer_y = customer_data['y_customer_id']
    
    # Get available terminals for this customer
    available_terminal_ids = customer_data['available_terminals']
    
    # Filter terminal data for available terminals
    available_terminals = terminal_profiles_table[
        terminal_profiles_table['TERMINAL_ID'].isin(available_terminal_ids)
    ].copy()
    
    # Calculate distances and classify as domestic/xb
    distances = []
    dom_xb_classification = []
    
    for _, terminal in available_terminals.iterrows():
        dist = np.sqrt((customer_x - terminal['x_terminal_id'])**2 + 
                      (customer_y - terminal['y_terminal_id'])**2)
        distances.append(dist)
        
        # Classification logic from your code
        if dist <= 0.90 * r:
            dom_xb_classification.append('domestic')
        else:
            dom_xb_classification.append('xb')
    
    available_terminals['distance'] = distances
    available_terminals['dom_xb'] = dom_xb_classification
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Spatial visualization
    # Plot all terminals in light gray
    ax1.scatter(terminal_profiles_table['x_terminal_id'], 
               terminal_profiles_table['y_terminal_id'], 
               c='lightgray', s=20, alpha=0.3, label='Other terminals')
    
    # Plot available terminals with domestic/xb coloring
    domestic_terminals = available_terminals[available_terminals['dom_xb'] == 'domestic']
    xb_terminals = available_terminals[available_terminals['dom_xb'] == 'xb']
    
    if len(domestic_terminals) > 0:
        ax1.scatter(domestic_terminals['x_terminal_id'], 
                   domestic_terminals['y_terminal_id'],
                   c='green', s=100, alpha=0.7, marker='s', 
                   label=f'Domestic terminals ({len(domestic_terminals)})')
    
    if len(xb_terminals) > 0:
        ax1.scatter(xb_terminals['x_terminal_id'], 
                   xb_terminals['y_terminal_id'],
                   c='red', s=100, alpha=0.7, marker='^', 
                   label=f'Cross-border terminals ({len(xb_terminals)})')
    
    # Plot customer
    ax1.scatter(customer_x, customer_y, c='blue', s=200, marker='*', 
               label=f'Customer {customer_id}', edgecolors='black', linewidth=2)
    
    # Add radius circles
    circle_full = Circle((customer_x, customer_y), r, fill=False, 
                        color='blue', linestyle='--', alpha=0.7, linewidth=2)
    circle_domestic = Circle((customer_x, customer_y), 0.90 * r, fill=False, 
                           color='green', linestyle=':', alpha=0.7, linewidth=2)
    
    ax1.add_patch(circle_full)
    ax1.add_patch(circle_domestic)
    
    ax1.set_xlim(-10, 510)
    ax1.set_ylim(-10, 510)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title(f'Customer {customer_id} Terminal Accessibility Map\n'
                 f'Radius: {r}, Domestic threshold: {0.90*r:.1f}')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Terminal analysis
    if show_mcc_info and len(available_terminals) > 0:
        # MCC code distribution
        mcc_counts = available_terminals.groupby(['MCC_CODE', 'dom_xb']).size().unstack(fill_value=0)
        
        # Create MCC code names mapping (from your original code)
        mcc_names = {
            5411: 'Grocery', 5541: 'Gas Stations', 5812: 'Restaurants',
            5311: 'Department', 5944: 'Jewelry', 5912: 'Drug Stores',
            5999: 'Misc Retail', 4121: 'Taxicabs', 5734: 'Software',
            5732: 'Electronics', 5691: 'Mens Clothing', 5651: 'Family Clothing',
            4411: 'Cruise Lines', 3000: 'Airlines', 4112: 'Railways',
            5921: 'Package Stores', 5993: 'Cigar', 7011: 'Hotels',
            5661: 'Shoe Stores', 5200: 'Home Supply'
        }
        
        # Add MCC names
        mcc_counts.index = [f"{mcc}\n{mcc_names.get(mcc, 'Unknown')}" for mcc in mcc_counts.index]
        
        # Plot stacked bar chart
        mcc_counts.plot(kind='bar', stacked=True, ax=ax2, 
                       color=['green', 'red'], alpha=0.7)
        ax2.set_title(f'Terminal Distribution by MCC Code\nCustomer {customer_id}')
        ax2.set_xlabel('MCC Code')
        ax2.set_ylabel('Number of Terminals')
        ax2.legend(['Domestic', 'Cross-border'])
        ax2.tick_params(axis='x', rotation=45)
        
    else:
        # Simple distance distribution
        ax2.hist(available_terminals['distance'], bins=20, alpha=0.7, 
                color='skyblue', edgecolor='black')
        ax2.axvline(x=0.90*r, color='green', linestyle=':', linewidth=2, 
                   label=f'Domestic threshold ({0.90*r:.1f})')
        ax2.axvline(x=r, color='blue', linestyle='--', linewidth=2, 
                   label=f'Max radius ({r})')
        ax2.set_xlabel('Distance from Customer')
        ax2.set_ylabel('Number of Terminals')
        ax2.set_title(f'Distance Distribution of Available Terminals\nCustomer {customer_id}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\n=== Customer {customer_id} Terminal Analysis ===")
    print(f"Customer location: ({customer_x:.1f}, {customer_y:.1f})")
    print(f"Total available terminals: {len(available_terminals)}")
    print(f"Domestic terminals: {len(domestic_terminals)}")
    print(f"Cross-border terminals: {len(xb_terminals)}")
    print(f"Customer spending profile:")
    print(f"  - Mean amount: ${customer_data['mean_amount']:.2f}")
    print(f"  - Std amount: ${customer_data['std_amount']:.2f}")
    print(f"  - Mean transactions per day: {customer_data['mean_nb_tx_per_day']:.2f}")
    
    if len(available_terminals) > 0:
        print(f"\nDistance statistics:")
        print(f"  - Min distance: {available_terminals['distance'].min():.1f}")
        print(f"  - Max distance: {available_terminals['distance'].max():.1f}")
        print(f"  - Mean distance: {available_terminals['distance'].mean():.1f}")
        
        print(f"\nMCC Code distribution:")
        mcc_summary = available_terminals.groupby('MCC_CODE').agg({
            'dom_xb': ['count', lambda x: (x=='domestic').sum(), lambda x: (x=='xb').sum()]
        }).round(2)
        mcc_summary.columns = ['Total', 'Domestic', 'Cross-border']
        
        for mcc in mcc_summary.index:
            mcc_name = mcc_names.get(mcc, 'Unknown')
            total = mcc_summary.loc[mcc, 'Total']
            domestic = mcc_summary.loc[mcc, 'Domestic']
            xb = mcc_summary.loc[mcc, 'Cross-border']
            print(f"  - {mcc} ({mcc_name}): {total} total ({domestic} domestic, {xb} xb)")
    
    plt.show()
    display(available_terminals.reset_index(drop=True))
    # return available_terminals