import os
import numpy as np
import pandas as pd
import datetime
import time
import random

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

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


def generate_transaction_response(tx_amount, mcc_code, tx_type, customer_id, terminal_id, random_seed):
    """
    Generate transaction response based on various factors
    Returns one of: 'Approved', 'Declined - Do not honor (DNH)', 'Declined - Not sufficient fund (NSF)', 
    'Declined - Life cycle', 'Declined - Call issuer', 'Declined - Policy', 'Declined - Technical', 'Declined - Security'
    
    Overall approval rate: 90%
    Decline reasons distribution (10% total declines):
    - NSF: 25% of declines (2.5% of total)
    - DNH: 20% of declines (2.0% of total) 
    - Call issuer: 15% of declines (1.5% of total)
    - Policy: 15% of declines (1.5% of total)
    - Security: 10% of declines (1.0% of total)
    - Technical: 8% of declines (0.8% of total)
    - Life cycle: 7% of declines (0.7% of total)
    """
    # Set random seed for reproducible results per transaction
    np.random.seed(random_seed)
    
    # Base approval probability
    approval_prob = 0.90
    
    # Adjust approval probability based on amount
    if tx_amount > 1000:
        approval_prob *= 0.95  # Slightly higher decline rate for very high amounts
    elif tx_amount > 500:
        approval_prob *= 0.97
    elif tx_amount < 5:
        approval_prob *= 0.98  # Very small amounts have higher approval
    
    # Adjust based on transaction type
    tx_type_adjustments = {
        'ecom': 0.98,       # E-commerce slightly higher decline
        'card-on-file': 0.99,
        'chip': 1.01,       # Chip transactions slightly more secure
        'tap': 1.01,        # Tap transactions slightly more secure
        'swipe': 1.00       # Baseline
    }
    approval_prob *= tx_type_adjustments.get(tx_type, 1.00)
    
    # Ensure probability stays within bounds
    approval_prob = max(0.85, min(0.95, approval_prob))  # Keep between 85% and 95%
    
    # Generate random number to determine approval
    rand_val = np.random.random()
    
    if rand_val < approval_prob:
        return 'Approved'
    else:
        # Transaction is declined, determine decline reason
        # Distribution of decline reasons (percentages are cumulative):
        decline_rand = np.random.random()
        
        if decline_rand < 0.25:  # 25% of declines
            return 'Declined - Not sufficient fund (NSF)'
        elif decline_rand < 0.45:  # 20% of declines (25% + 20%)
            return 'Declined - Do not honor (DNH)'
        elif decline_rand < 0.60:  # 15% of declines (45% + 15%)
            return 'Declined - Call issuer'
        elif decline_rand < 0.75:  # 15% of declines (60% + 15%)
            return 'Declined - Policy'
        elif decline_rand < 0.85:  # 10% of declines (75% + 10%)
            return 'Declined - Security'
        elif decline_rand < 0.93:  # 8% of declines (85% + 8%)
            return 'Declined - Technical'
        else:  # 7% of declines (remaining)
            return 'Declined - Life cycle'


def generate_transactions_table(customer_profile, terminal_profiles_table, start_date, nb_days, r):
    """
    Generate transactions table with MCC codes, transaction types, and responses
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
                        
                        # Generate transaction response
                        # Create a unique seed for each transaction
                        tx_seed = int(customer_profile.CUSTOMER_ID * 10000 + day * 100 + tx + time_tx % 1000)
                        tx_response = generate_transaction_response(
                            amount, mcc_code, tx_type, 
                            customer_profile.CUSTOMER_ID, terminal_id, tx_seed
                        )
                        
                        customer_transactions.append([time_tx + day * 86400, day,
                                                    customer_profile.CUSTOMER_ID,
                                                    terminal_id, amount, mcc_code, tx_type, dom_xb, tx_response])
    
    customer_transactions = pd.DataFrame(customer_transactions, 
                                       columns=['TX_TIME_SECONDS', 'TX_TIME_DAYS', 
                                               'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT',
                                               'MCC_CODE', 'TX_TYPE', 'TX_DOM_XB', 'TX_RESPONSE'])
    
    if len(customer_transactions) > 0:
        customer_transactions['TX_DATETIME'] = pd.to_datetime(customer_transactions["TX_TIME_SECONDS"], 
                                                            unit='s', origin=start_date)
        customer_transactions = customer_transactions[['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 
                                                     'TX_AMOUNT', 'MCC_CODE', 'TX_TYPE', 'TX_DOM_XB', 'TX_RESPONSE',
                                                     'TX_TIME_SECONDS', 'TX_TIME_DAYS']]
    
    return customer_transactions


def add_frauds(customer_profiles_table: pd.DataFrame, 
               terminal_profiles_table: pd.DataFrame, 
               transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sophisticated fraud scenarios with realistic patterns
    IMPORTANT: Only approved transactions can be marked as fraud
    
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
    
    # Only consider approved transactions for fraud marking
    approved_mask = transactions_df['TX_RESPONSE'] == 'Approved'
    approved_transactions = transactions_df[approved_mask]
    
    print(f"Total transactions: {len(transactions_df)}")
    print(f"Approved transactions: {len(approved_transactions)} ({len(approved_transactions)/len(transactions_df)*100:.1f}%)")
    
    if len(approved_transactions) == 0:
        print("No approved transactions to mark as fraud")
        return transactions_df
    
    # Scenario 1: High-value transactions (MCC-aware) - ONLY APPROVED
    high_value_mccs = {5944: 500, 5734: 300, 4411: 800, 3000: 400, 7011: 350}
    
    fraud_count_s1 = 0
    for mcc, threshold in high_value_mccs.items():
        mask = (approved_mask & 
                (transactions_df.MCC_CODE == mcc) & 
                (transactions_df.TX_AMOUNT > threshold))
        transactions_df.loc[mask, 'TX_FRAUD'] = 1
        transactions_df.loc[mask, 'TX_FRAUD_SCENARIO'] = 1
        fraud_count_s1 += mask.sum()
    
    # Other MCCs with general threshold - ONLY APPROVED
    other_mask = (approved_mask & 
                  (~transactions_df.MCC_CODE.isin(high_value_mccs.keys())) & 
                  (transactions_df.TX_AMOUNT > 220))
    transactions_df.loc[other_mask, 'TX_FRAUD'] = 1
    transactions_df.loc[other_mask, 'TX_FRAUD_SCENARIO'] = 1
    fraud_count_s1 += other_mask.sum()
    
    print(f"✓ Scenario 1 (High-value): {fraud_count_s1} frauds")
    
    # Scenario 2: Compromised terminals - ONLY APPROVED
    fraud_count_s2 = 0
    max_days = transactions_df.TX_TIME_DAYS.max()
    
    for day in range(0, max_days, 28):  # Every 4 weeks
        n_compromised = min(2, len(terminal_profiles_table))
        compromised_terminals = terminal_profiles_table.TERMINAL_ID.sample(
            n=n_compromised, random_state=day
        ).values
        
        # Transactions at compromised terminals - ONLY APPROVED
        mask = (approved_mask &
                (transactions_df.TX_TIME_DAYS >= day) &
                (transactions_df.TX_TIME_DAYS < day + 28) &
                (transactions_df.TERMINAL_ID.isin(compromised_terminals)))
        
        compromised_tx = transactions_df[mask]
        
        # Transaction type fraud probabilities
        fraud_probs = {'swipe': 0.3, 
                       'chip': 0.4, 
                       'card-on-file': 0.6, 
                       'ecom': 0.8, 
                       'tap': 0.2}
        
        for idx in compromised_tx.index:
            tx_type = transactions_df.loc[idx, 'TX_TYPE']
            if np.random.random() < fraud_probs.get(tx_type, 0.5):
                transactions_df.loc[idx, 'TX_FRAUD'] = 1
                transactions_df.loc[idx, 'TX_FRAUD_SCENARIO'] = 2
                fraud_count_s2 += 1
    
    print(f"✓ Scenario 2 (Compromised terminals): {fraud_count_s2} frauds")
    
    # Scenario 3: Compromised cards - ONLY APPROVED
    fraud_count_s3 = 0
    
    for day in range(0, max_days, 21):  # Every 3 weeks
        n_compromised = min(5, len(customer_profiles_table))
        compromised_customers = customer_profiles_table.CUSTOMER_ID.sample(
            n=n_compromised, random_state=day
        ).values
        
        # Transactions by compromised customers - ONLY APPROVED
        mask = (approved_mask &
                (transactions_df.TX_TIME_DAYS >= day) &
                (transactions_df.TX_TIME_DAYS < day + 14) &
                (transactions_df.CUSTOMER_ID.isin(compromised_customers)))
        
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

    # Scenario 4: Account Takeover - VERY RARE - ONLY APPROVED
    fraud_count_s4 = 0
    
    # Very rare ATO attacks (every 120-180 days)
    ato_intervals = [120, 150, 180, 135, 165, 140, 175, 160, 145, 170]
    current_day = 0
    interval_idx = 0
    
    while current_day < max_days:
        # Only 1 customer per ATO attack (was 1-2)
        n_ato_customers = 2
        ato_customers = customer_profiles_table.CUSTOMER_ID.sample(
            n=n_ato_customers, random_state=current_day
        ).values
        
        # Very short ATO window: 1-3 days (was 2-5)
        ato_duration = np.random.randint(1, 4)
        
        mask = (approved_mask &
                (transactions_df.TX_TIME_DAYS >= current_day) &
                (transactions_df.TX_TIME_DAYS < current_day + ato_duration) &
                (transactions_df.CUSTOMER_ID.isin(ato_customers)))
        
        ato_tx = transactions_df[mask]
        
        if len(ato_tx) > 0:
            # ATO patterns: Only very high-value, CNP transactions
            high_value_ato = ato_tx[ato_tx.TX_AMOUNT > ato_tx.TX_AMOUNT.quantile(0.95)]  # Top 5% only
            cnp_ato = ato_tx[ato_tx.TX_TYPE.isin(['ecom', 'card-on-file'])]
            
            # Only select transactions that are BOTH high-value AND CNP
            potential_ato_frauds = high_value_ato[high_value_ato.TX_TYPE.isin(['ecom', 'card-on-file'])]
            
            if len(potential_ato_frauds) > 0:
                # Select only 30-50% of potential ATO transactions (was 70-90%)
                ato_fraud_rate = np.random.uniform(0.30, 0.50)
                n_ato_frauds = max(1, int(len(potential_ato_frauds) * ato_fraud_rate))
                
                ato_fraud_indices = np.random.choice(
                    potential_ato_frauds.index, 
                    min(n_ato_frauds, len(potential_ato_frauds)), 
                    replace=False
                )
                
                for idx in ato_fraud_indices:
                    # ATO frauds: smaller amount increase (2-4x instead of 4-10x)
                    transactions_df.loc[idx, 'TX_AMOUNT'] *= np.random.uniform(2, 4)
                    transactions_df.loc[idx, 'TX_FRAUD'] = 1
                    transactions_df.loc[idx, 'TX_FRAUD_SCENARIO'] = 4
                    fraud_count_s4 += 1
        
        # Move to next ATO event
        current_day += ato_intervals[interval_idx % len(ato_intervals)]
        interval_idx += 1
    
    print(f"✓ Scenario 4 (Account Takeover): {fraud_count_s4} frauds")
    
    total_frauds = transactions_df.TX_FRAUD.sum()
    approved_count = len(approved_transactions)
    fraud_rate_of_approved = total_frauds / approved_count * 100 if approved_count > 0 else 0
    fraud_rate_of_total = total_frauds / len(transactions_df) * 100
    
    print(f"✓ Total frauds: {total_frauds} ({fraud_rate_of_approved:.2f}% of approved, {fraud_rate_of_total:.2f}% of all transactions)")

    # Print scenario breakdown
    if 'TX_FRAUD_SCENARIO' in transactions_df.columns:
        print(f"\nFraud Scenario Breakdown:")
        fraud_scenarios = transactions_df[transactions_df.TX_FRAUD == 1].TX_FRAUD_SCENARIO.value_counts().sort_index()
        scenario_names = {
            1: "High-value transactions",
            2: "Compromised terminals", 
            3: "Compromised cards",
            4: "Account takeover"
        }
        
        for scenario, count in fraud_scenarios.items():
            scenario_name = scenario_names.get(scenario, f"Scenario {scenario}")
            pct = count / total_frauds * 100 if total_frauds > 0 else 0
            print(f"  {scenario_name}: {count:,} ({pct:.1f}%)")
    
    # Print response breakdown
    print(f"\nTransaction Response Breakdown:")
    response_counts = transactions_df.TX_RESPONSE.value_counts()
    for response, count in response_counts.items():
        pct = count / len(transactions_df) * 100
        print(f"  {response}: {count:,} ({pct:.1f}%)")
    
    # Verify no declined transactions are marked as fraud
    declined_frauds = transactions_df[(transactions_df.TX_RESPONSE != 'Approved') & (transactions_df.TX_FRAUD == 1)]
    if len(declined_frauds) > 0:
        print(f"\n⚠️  WARNING: {len(declined_frauds)} declined transactions incorrectly marked as fraud!")
    else:
        print(f"\n✓ Verification passed: No declined transactions marked as fraud")
    
    return transactions_df