import numpy as np
import pandas as pd

def calculate_normal_operational_limits(df, feature_store, is_training=True):
    """
    Calculate actual operational limits from normal data (not fixed percentages)
    """
    if not is_training:
        return feature_store
    
    print("\n=== Calculating Normal Operational Limits from Data ===")
    
    N_BUSES = 57
    N_HOURS = 24
    FEATURES_PER_HOUR = 270
    
    # Use only normal data
    normal_df = df[df['attack'] == 0].copy()
    
    generator_buses = feature_store.get('generator_buses', [])
    generator_operational_limits = {}
    
    for gen_bus in generator_buses:
        gen_bus_idx = gen_bus - 1  # Convert to 0-based
        
        # Collect all power values across all hours
        all_power_values = []
        for hour in range(N_HOURS):
            col = f'ft{hour * FEATURES_PER_HOUR + gen_bus_idx + 1}'
            if col in normal_df.columns:
                power_values = normal_df[col].values
                # Only consider non-zero values for operational limits
                non_zero_values = power_values[power_values > 0.001]
                if len(non_zero_values) > 0:
                    all_power_values.extend(non_zero_values)
        
        if all_power_values:
            # Use percentiles from actual normal operation
            generator_operational_limits[gen_bus] = {
                'upper_normal': np.percentile(all_power_values, 95),  # 95th percentile as normal upper
                'upper_extreme': np.percentile(all_power_values, 99),  # 99th percentile as extreme
                'lower_normal': np.percentile(all_power_values, 5),   # 5th percentile as normal lower
                'lower_extreme': np.percentile(all_power_values, 1),   # 1st percentile as extreme
                'median': np.median(all_power_values),
                'mean': np.mean(all_power_values)
            }
            
            if gen_bus in generator_buses[:5]:  # Print for top 5 generators
                print(f"Generator Bus {gen_bus}:")
                print(f"  Upper normal (95%): {generator_operational_limits[gen_bus]['upper_normal']:.2f} MW")
                print(f"  Upper extreme (99%): {generator_operational_limits[gen_bus]['upper_extreme']:.2f} MW")
                print(f"  Lower normal (5%): {generator_operational_limits[gen_bus]['lower_normal']:.2f} MW")
                print(f"  Lower extreme (1%): {generator_operational_limits[gen_bus]['lower_extreme']:.2f} MW")
    
    feature_store['generator_operational_limits'] = generator_operational_limits
    
    return feature_store

def calculate_bidirectional_ramp_limits(df, feature_store, is_training=True):
    """Calculate BOTH positive (ramp-up) and negative (ramp-down) limits from normal data"""
    if not is_training:
        return feature_store
    
    print("\n=== Calculating Bidirectional Ramp Limits from Data ===")
    
    N_BUSES = 57
    N_HOURS = 24 
    FEATURES_PER_HOUR = 270
    
    # Use only normal data
    normal_df = df[df['attack'] == 0].copy()
    
    generator_buses = feature_store.get('generator_buses', [])
    generator_ramp_limits = {}
    
    # Track statistics
    no_ramp_generators = []
    unidirectional_generators = []
    bidirectional_generators = []
    
    for gen_bus in generator_buses:
        gen_bus_idx = gen_bus - 1
        
        # Collect power values across all hours
        gen_profile = []
        for hour in range(N_HOURS):
            col = f'ft{hour * FEATURES_PER_HOUR + gen_bus_idx + 1}'
            if col in normal_df.columns:
                gen_profile.append(normal_df[col])
        
        if len(gen_profile) > 1:
            gen_profile_df = pd.concat(gen_profile, axis=1)
            
            # Calculate hour-to-hour ramps
            ramps = gen_profile_df.diff(axis=1)
            
            # Separate positive and negative ramps
            positive_ramps = ramps[ramps > 0].values.flatten()
            negative_ramps = ramps[ramps < 0].values.flatten()
            
            positive_ramps = positive_ramps[~np.isnan(positive_ramps)]
            negative_ramps = negative_ramps[~np.isnan(negative_ramps)]
            
            # Categorize generator
            if len(positive_ramps) == 0 and len(negative_ramps) == 0:
                no_ramp_generators.append(gen_bus)
                # Store zero ramps for constant generators
                generator_ramp_limits[gen_bus] = {
                    'ramp_up_normal': 0,
                    'ramp_up_extreme': 0,
                    'ramp_up_mean': 0,
                    'ramp_up_std': 0,
                    'ramp_down_normal': 0,
                    'ramp_down_extreme': 0,
                    'ramp_down_mean': 0,
                    'ramp_down_std': 0,
                    'type': 'constant'
                }
            elif len(positive_ramps) > 0 and len(negative_ramps) == 0:
                unidirectional_generators.append((gen_bus, 'up_only'))
                # Store only upward ramps
                generator_ramp_limits[gen_bus] = {
                    'ramp_up_normal': np.percentile(positive_ramps, 95),
                    'ramp_up_extreme': np.percentile(positive_ramps, 99),
                    'ramp_up_mean': np.mean(positive_ramps),
                    'ramp_up_std': np.std(positive_ramps),
                    'ramp_down_normal': 0,
                    'ramp_down_extreme': 0,
                    'ramp_down_mean': 0,
                    'ramp_down_std': 0,
                    'type': 'up_only'
                }
            elif len(positive_ramps) == 0 and len(negative_ramps) > 0:
                unidirectional_generators.append((gen_bus, 'down_only'))
                # Store only downward ramps
                generator_ramp_limits[gen_bus] = {
                    'ramp_up_normal': 0,
                    'ramp_up_extreme': 0,
                    'ramp_up_mean': 0,
                    'ramp_up_std': 0,
                    'ramp_down_normal': np.percentile(np.abs(negative_ramps), 95),
                    'ramp_down_extreme': np.percentile(np.abs(negative_ramps), 99),
                    'ramp_down_mean': np.mean(np.abs(negative_ramps)),
                    'ramp_down_std': np.std(np.abs(negative_ramps)),
                    'type': 'down_only'
                }
            else:
                bidirectional_generators.append(gen_bus)
                # Store both ramps 
                generator_ramp_limits[gen_bus] = {
                    'ramp_up_normal': np.percentile(positive_ramps, 95),
                    'ramp_up_extreme': np.percentile(positive_ramps, 99),
                    'ramp_up_mean': np.mean(positive_ramps),
                    'ramp_up_std': np.std(positive_ramps),
                    'ramp_down_normal': np.percentile(np.abs(negative_ramps), 95),
                    'ramp_down_extreme': np.percentile(np.abs(negative_ramps), 99),
                    'ramp_down_mean': np.mean(np.abs(negative_ramps)),
                    'ramp_down_std': np.std(np.abs(negative_ramps)),
                    'type': 'bidirectional'
                }
    
    # Print summary
    print(f"\nGenerator Ramping Summary:")
    print(f"  Constant (no ramping): {len(no_ramp_generators)} generators")
    print(f"  Unidirectional ramping: {len(unidirectional_generators)} generators")
    print(f"  Bidirectional ramping: {len(bidirectional_generators)} generators")
    
    # Print details for first 5 of each type
    if no_ramp_generators:
        print(f"\nConstant generators (first 5): {no_ramp_generators[:5]}")
    
    if bidirectional_generators:
        print(f"\nBidirectional ramping generators (showing first 5):")
        for gen_bus in bidirectional_generators[:5]:
            limits = generator_ramp_limits[gen_bus]
            print(f"  Generator Bus {gen_bus}:")
            print(f"    Ramp UP normal (95%): {limits['ramp_up_normal']:.2f} MW/h")
            print(f"    Ramp DOWN normal (95%): {limits['ramp_down_normal']:.2f} MW/h")
    
    feature_store['generator_ramp_limits'] = generator_ramp_limits
    feature_store['constant_generators'] = no_ramp_generators
    feature_store['ramping_generators'] = bidirectional_generators + [g[0] for g in unidirectional_generators]
    
    return feature_store

def create_improved_knn_models(df, feature_store, is_training=True):
    # import necessary libraries
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    """
    Create improved k-NN models with:
    1. Feature scaling
    2. Leave-one-out for training data (n_neighbors=2)
    3. Percentile-based thresholds
    """
    print("\n=== Creating Improved k-NN Models for Generator Anomaly Detection ===")
    
    generator_buses = feature_store.get('generator_buses', [])   
    # Get generator-specific features
    top_generators = generator_buses
    gen_features = [col for col in df.columns if any(f'bus{gb}_' in col for gb in top_generators)]
    
    # Initialize columns
    df['gen_anomaly_knn_distance'] = 0.0
    
    
    if is_training and 'attack' in df.columns:
        print(f"Building k-NN models using {len(gen_features)} generator features...")
        
        knn_models = {}
        knn_thresholds = {}
        scalers = {}
        
        for ssn in df['original_ssn'].unique():
            ssn_mask = df['original_ssn'] == ssn
            ssn_normal_mask = ssn_mask & (df['attack'] == 0)
            ssn_normal_data = df[ssn_normal_mask]
            
            if len(ssn_normal_data) >= 10:  # Need sufficient normal samples
                print(f"  Building k-NN for SSN {ssn} with {len(ssn_normal_data)} normal samples")
                
                # Scale features
                scaler = StandardScaler()
                normal_features_scaled = scaler.fit_transform(ssn_normal_data[gen_features])
                
                # Build k-NN model
                # Use min(5, half of normal samples) for k
                k = min(5, len(ssn_normal_data) // 2)
                knn_model = NearestNeighbors(n_neighbors=k, metric='manhattan')
                knn_model.fit(normal_features_scaled)
                
                # Calculate leave-one-out distances for normal data
                # Use n_neighbors=2 to avoid self-matches
                distances_loo, _ = knn_model.kneighbors(normal_features_scaled, n_neighbors=2)
                # Take the second nearest (first is self)
                normal_distances = distances_loo[:, 1]
                
                # Store models and thresholds
                knn_models[ssn] = knn_model
                scalers[ssn] = scaler
                knn_thresholds[ssn] = {
                    'median_normal': np.median(normal_distances),
                    'mean_normal': np.mean(normal_distances)
                }
                
                # Apply to all data for this SSN
                ssn_all_data = df[ssn_mask]
                all_features_scaled = scaler.transform(ssn_all_data[gen_features])
                
                # For training data: use appropriate neighbor count
                # For normal samples in training, use n_neighbors=2 (leave-one-out)
                # For attack samples, use n_neighbors=1 (nearest normal)
                distances_all = np.zeros(len(ssn_all_data))
                
                for idx, (_, row) in enumerate(ssn_all_data.iterrows()):
                    if row.get('attack', 1) == 0:  # Normal sample
                        # Leave-one-out: find 2nd nearest
                        dist, _ = knn_model.kneighbors(all_features_scaled[idx:idx+1], n_neighbors=2)
                        distances_all[idx] = dist[0, 1]  # 2nd nearest
                    else:  # Attack sample or test data
                        # Find nearest normal
                        dist, _ = knn_model.kneighbors(all_features_scaled[idx:idx+1], n_neighbors=1)
                        distances_all[idx] = dist[0, 0]
                
                df.loc[ssn_mask, 'gen_anomaly_knn_distance'] = distances_all
               
        
        feature_store['knn_models'] = knn_models
        feature_store['knn_scalers'] = scalers
        feature_store['knn_features'] = gen_features
        
        # Print summary statistics
        if 'attack' in df.columns:
            normal_distances = df[df['attack'] == 0]['gen_anomaly_knn_distance']
            attack_distances = df[df['attack'] == 1]['gen_anomaly_knn_distance']
            
            print(f"\n  Distance Statistics:")
            print(f"    Normal samples - Mean: {normal_distances.mean():.3f}, Median: {normal_distances.median():.3f}")
            print(f"    Attack samples - Mean: {attack_distances.mean():.3f}, Median: {attack_distances.median():.3f}")
           
    
    else:  # Test mode
        if all(key in feature_store for key in ['knn_models', 'knn_scalers','knn_features']):
            print("Applying k-NN models to test data...")
            
            knn_models = feature_store['knn_models']
            scalers = feature_store['knn_scalers']
            knn_features = feature_store['knn_features']
            
            # Check feature availability
            valid_features = [f for f in knn_features if f in df.columns]

            
            for ssn in knn_models.keys():
                if ssn not in df['original_ssn'].unique():
                    continue
                    
                ssn_mask = df['original_ssn'] == ssn
                ssn_data = df[ssn_mask]
                
                # Scale features
                features_scaled = scalers[ssn].transform(ssn_data[valid_features])
                
                # Calculate distances to nearest normal
                distances, _ = knn_models[ssn].kneighbors(features_scaled, n_neighbors=1)
                distances = distances.ravel()
                
               
                
                df.loc[ssn_mask, 'gen_anomaly_knn_distance'] = distances
               
            print(f"  Applied k-NN to {len(knn_models)} SSNs")
        
    
    return df, feature_store


def calculate_normal_flow_and_angle_baselines(df, feature_store=None, is_training=True):
    """
    Calculate normal behavior baselines for flows and angles per SSN/hour
    """
    N_BUSES = 57
    N_HOURS = 24
    FEATURES_PER_HOUR = 270
    N_FLOWS = 156
    
    if feature_store is None:
        feature_store = {}
    
    
    
    if is_training:
        print("\n=== Calculating Normal Behavior Baselines ===")
        # Only use normal data for baseline
        normal_df = df[df['attack'] == 0].copy()
        
        # Store baselines for flows and angles
        flow_baselines = {}
        angle_baselines = {}
        
        for ssn in normal_df['original_ssn'].unique():
            ssn_normal = normal_df[normal_df['original_ssn'] == ssn]
            
            flow_baselines[ssn] = {}
            angle_baselines[ssn] = {}
            
            for hour in range(N_HOURS):
                # Power flows 
                flow_start = hour * FEATURES_PER_HOUR + 2 * N_BUSES
                flow_cols = [f'ft{flow_start + i + 1}' for i in range(N_FLOWS)]
                flow_cols = [col for col in flow_cols if col in ssn_normal.columns]
                
                if flow_cols:
                    # Calculate mean and std for each flow in this hour
                    flow_means = ssn_normal[flow_cols].mean()
                    flow_stds = ssn_normal[flow_cols].std()
                    flow_baselines[ssn][hour] = {
                        'means': flow_means.to_dict(),
                        'stds': flow_stds.to_dict()
                    }
                
                # Voltage angles 
                angle_start = hour * FEATURES_PER_HOUR + N_BUSES
                angle_cols = [f'ft{angle_start + i + 1}' for i in range(N_BUSES)]
                angle_cols = [col for col in angle_cols if col in ssn_normal.columns]
                
                if angle_cols:
                    angle_means = ssn_normal[angle_cols].mean()
                    angle_stds = ssn_normal[angle_cols].std()
                    angle_baselines[ssn][hour] = {
                        'means': angle_means.to_dict(),
                        'stds': angle_stds.to_dict()
                    }
        
        feature_store['flow_baselines'] = flow_baselines
        feature_store['angle_baselines'] = angle_baselines
        
        print(f"Calculated baselines for {len(flow_baselines)} SSNs")
    
    # Create deviation features for all data
    flow_baselines = feature_store.get('flow_baselines', {})
    angle_baselines = feature_store.get('angle_baselines', {})
    
    # Initialize deviation features
    for hour in range(N_HOURS):  # all hours
        df[f'flow_deviation_h{hour}_mean'] = 0.0
        df[f'flow_deviation_h{hour}_max'] = 0.0
        df[f'angle_deviation_h{hour}_mean'] = 0.0
        df[f'angle_deviation_h{hour}_max'] = 0.0
    
    for ssn in df['original_ssn'].unique():
        
            
        ssn_mask = df['original_ssn'] == ssn
        ssn_data = df[ssn_mask]
        
        for hour in range(N_HOURS):
            # Flow deviations
            if hour in flow_baselines[ssn]:
                flow_start = hour * FEATURES_PER_HOUR + 2 * N_BUSES
                flow_cols = [f'ft{flow_start + i + 1}' for i in range(N_FLOWS)]
                
                deviations = []
                for col in flow_cols:
                    if col in ssn_data.columns and col in flow_baselines[ssn][hour]['means']:
                        mean_val = flow_baselines[ssn][hour]['means'][col]
                        std_val = flow_baselines[ssn][hour]['stds'][col]
                        if std_val > 0:
                            # Z-score deviation
                            deviation = (ssn_data[col] - mean_val) / std_val
                            deviations.append(deviation.abs())
                
                if deviations:
                    deviations_df = pd.concat(deviations, axis=1)
                    df.loc[ssn_mask, f'flow_deviation_h{hour}_mean'] = deviations_df.mean(axis=1)
                    df.loc[ssn_mask, f'flow_deviation_h{hour}_max'] = deviations_df.max(axis=1)
            
            # Angle deviations
            if hour in angle_baselines[ssn]:
                angle_start = hour * FEATURES_PER_HOUR + N_BUSES
                angle_cols = [f'ft{angle_start + i + 1}' for i in range(N_BUSES)]
                
                deviations = []
                for col in angle_cols:
                    if col in ssn_data.columns and col in angle_baselines[ssn][hour]['means']:
                        mean_val = angle_baselines[ssn][hour]['means'][col]
                        std_val = angle_baselines[ssn][hour]['stds'][col]
                        if std_val > 0:
                            deviation = (ssn_data[col] - mean_val) / std_val
                            deviations.append(deviation.abs())
                
                if deviations:
                    deviations_df = pd.concat(deviations, axis=1)
                    df.loc[ssn_mask, f'angle_deviation_h{hour}_mean'] = deviations_df.mean(axis=1)
                    df.loc[ssn_mask, f'angle_deviation_h{hour}_max'] = deviations_df.max(axis=1)
    
    return df, feature_store




def identify_generator_buses(df, feature_store=None, verbose=True):
    """
    Identify generators using ONLY NORMAL DATA to establish proper merit order
    """
    N_BUSES = 57
    N_HOURS = 24
    FEATURES_PER_HOUR = 270

    if feature_store is None:
        feature_store = {}

    print("=" * 60)
    print("IDENTIFYING GENERATOR BUSES FROM NORMAL DATA ONLY")
    print("=" * 60)

    # CRITICAL: Use only normal data to identify generators
    if 'attack' in df.columns:
        normal_df = df[df['attack'] == 0].copy()
        print(f"Using {len(normal_df)} normal samples to identify generators")
    else:
        normal_df = df.copy()
        print("Warning: No attack column found, using all data")

    # Arrays for tracking stats
    bus_generation_hours = np.zeros(N_BUSES)
    bus_max_generation = np.zeros(N_BUSES)
    bus_avg_generation = np.zeros(N_BUSES)
    bus_capacity_factor = np.zeros(N_BUSES)

    # Loop over buses
    for bus_idx in range(N_BUSES):
        positive_hours = 0
        max_gen = 0
        all_values = []

        for hour in range(N_HOURS):
            col = f'ft{hour * FEATURES_PER_HOUR + bus_idx + 1}'
            if col in normal_df.columns:
                power_values = normal_df[col].values
                positive_mask = power_values > 0.001

                if positive_mask.any():
                    positive_hours += 1
                    max_gen = max(max_gen, power_values[positive_mask].max())
                    all_values.extend(power_values[positive_mask])

        if all_values:
            bus_avg_generation[bus_idx] = np.mean(all_values)
            bus_capacity_factor[bus_idx] = bus_avg_generation[bus_idx] / (max_gen + 1e-6)
        
        bus_generation_hours[bus_idx] = positive_hours
        bus_max_generation[bus_idx] = max_gen

  
    
    # Identify generators
    generator_threshold = 1 # Any hour with generation counts
    generator_mask = bus_generation_hours >= generator_threshold
    generator_buses_idx = np.where(generator_mask)[0]
    

    # CRITICAL: Sort by average generation (proxy for marginal cost)
    # Larger average generation usually means baseload/cheaper units
    generator_info = [
        (idx, bus_max_generation[idx], bus_avg_generation[idx], 
         bus_generation_hours[idx], bus_capacity_factor[idx])
        for idx in generator_buses_idx
    ]
    
    # Sort by average generation (merit order proxy)
    generator_info.sort(key=lambda x: x[2], reverse=True)

    # Store results
    feature_store['generator_buses'] = [g[0] + 1 for g in generator_info]
    feature_store['generator_capacities'] = {g[0] + 1: g[1] for g in generator_info}
    feature_store['generator_avg_gen'] = {g[0] + 1: g[2] for g in generator_info}
    feature_store['generator_capacity_factors'] = {g[0] + 1: g[4] for g in generator_info}
    
    # Store merit order based on normal operation
    feature_store['merit_order'] = [g[0] + 1 for g in generator_info]

    if verbose:
        print(f"\nFound {len(generator_buses_idx)} GENERATORS (from normal data):")
        print("-" * 70)
        print(f"{'Merit':<6} {'Bus ID':<10} {'Capacity':<12} {'Avg Gen':<12} {'Cap Factor':<12} {'Type'}")
        print("-" * 70)

        for i, (bus_idx, capacity, avg_gen, hours, cap_factor) in enumerate(generator_info[:20]):
            real_bus_id = bus_idx + 1
            gen_type = "Base Load" if cap_factor >= 0.7 else "Peaking" if cap_factor <= 0.3 else "Intermediate"
            print(f"{i+1:<6} Bus {real_bus_id:<6} {capacity:>10.2f} MW {avg_gen:>10.2f} MW {cap_factor:>10.2%} {gen_type}")

    return feature_store


def engineer_generator_focused_features(df, is_training=False, feature_store=None):
    """
    Refactored generator feature engineering - cleaner, no redundancy
    """
    import numpy as np
    import pandas as pd
    from scipy import stats
    
    if feature_store is None:
        feature_store = {}
    
    N_BUSES = 57
    N_HOURS = 24
    FEATURES_PER_HOUR = 270
    
    generator_buses = feature_store.get('generator_buses', [])
    generator_capacities = feature_store.get('generator_capacities', {})
    generator_avg_gen = feature_store.get('generator_avg_gen', {})
    generator_capacity_factors = feature_store.get('generator_capacity_factors', {})
    generator_ramp_limits = feature_store.get('generator_ramp_limits', {})
    generator_operational_limits = feature_store.get('generator_operational_limits', {})
    merit_order = feature_store.get('merit_order', [])
    
    print(f"\nCreating features for {len(generator_buses)} identified generators...")
    
   
    all_generators = [g for g in generator_buses]
    print(f"Processing {len(all_generators)} generators")
    
    # ============ INDIVIDUAL GENERATOR FEATURES ============
    for gen_rank, gen_bus in enumerate(all_generators):
        # Build 24-hour profile
        gen_powers = []
        for hour in range(N_HOURS):
            col = f'ft{hour * FEATURES_PER_HOUR + (gen_bus-1) + 1}'
            if col in df.columns:
                gen_powers.append(df[col])
        
        if not gen_powers:
            continue
     
    # # Track top 10 largest generators in detail
    # top_generators = generator_buses[:min(10, len(generator_buses))]
    # print(f"Focusing on top {len(top_generators)} generators: {top_generators}")
    
    # for gen_rank, gen_bus in enumerate(top_generators):
    #     gen_powers = []
        
    #     for hour in range(N_HOURS):
    #         col = f'ft{hour * FEATURES_PER_HOUR + (gen_bus-1) + 1}'  # gen_bus is 1-based
    #         if col in df.columns:
    #             gen_powers.append(df[col])
        
        gen_power_df = pd.concat(gen_powers, axis=1)
        
        
        
        # Get generator parameters
        gen_capacity = generator_capacities.get(gen_bus, gen_power_df.max().max())
        gen_avg = generator_avg_gen.get(gen_bus, gen_power_df.mean().mean())
        
        # Get operational limits from data
        if gen_bus in generator_operational_limits:
            limits = generator_operational_limits[gen_bus]
            upper_normal = limits['upper_normal']
            upper_extreme = limits['upper_extreme']
            lower_normal = limits['lower_normal']
            lower_extreme = limits['lower_extreme']
        else:
            # Fallback to percentage-based
            upper_normal = gen_capacity * 0.95
            upper_extreme = gen_capacity * 0.99
            lower_normal = gen_capacity * 0.05
            lower_extreme = gen_capacity * 0.01
        
        # Get ramp limits
        if gen_bus in generator_ramp_limits:
            ramp_limits = generator_ramp_limits[gen_bus]
            ramp_up_normal = ramp_limits.get('ramp_up_normal', gen_capacity * 0.2)
            ramp_up_extreme = ramp_limits.get('ramp_up_extreme', gen_capacity * 0.4)
            ramp_down_normal = ramp_limits.get('ramp_down_normal', gen_capacity * 0.2)
            ramp_down_extreme = ramp_limits.get('ramp_down_extreme', gen_capacity * 0.4)
        else:
            ramp_up_normal = ramp_down_normal = gen_capacity * 0.2
            ramp_up_extreme = ramp_down_extreme = gen_capacity * 0.4
        
        # ===== CATEGORY 1: BASIC STATISTICS =====
        df[f'gen_{gen_rank}_bus{gen_bus}_mean'] = gen_power_df.mean(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_std'] = gen_power_df.std(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_max'] = gen_power_df.max(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_min'] = gen_power_df.min(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_range'] = gen_power_df.max(axis=1) - gen_power_df.min(axis=1)
        
        # Capacity metrics
        df[f'gen_{gen_rank}_bus{gen_bus}_capacity_factor'] = (
            gen_power_df.mean(axis=1) / (gen_capacity + 1e-6)
        )
        df[f'gen_{gen_rank}_bus{gen_bus}_max_utilization'] = (
            gen_power_df.max(axis=1) / (gen_capacity + 1e-6)
        )
        df[f'gen_{gen_rank}_bus{gen_bus}_deviation_from_normal'] = (
            (gen_power_df.mean(axis=1) - gen_avg) / (gen_avg + 1e-6)
        )
        
        # ===== CATEGORY 2: RAMP FEATURES (ALL IN ONE PLACE) =====
        ramps = gen_power_df.diff(axis=1)
        positive_ramps = ramps.clip(lower=0)
        negative_ramps = ramps.clip(upper=0)
        
        # Basic ramp stats
        df[f'gen_{gen_rank}_bus{gen_bus}_max_ramp_up'] = positive_ramps.max(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_mean_ramp_up'] = positive_ramps[positive_ramps > 0].mean(axis=1).fillna(0)
        df[f'gen_{gen_rank}_bus{gen_bus}_max_ramp_down'] = negative_ramps.abs().max(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_mean_ramp_down'] = negative_ramps[negative_ramps < 0].abs().mean(axis=1).fillna(0)
        df[f'gen_{gen_rank}_bus{gen_bus}_ramp_std'] = ramps.std(axis=1)
        
        # Normalized ramps
        df[f'gen_{gen_rank}_bus{gen_bus}_ramp_up_normalized'] = (
            positive_ramps.max(axis=1) / (ramp_up_normal + 1e-6)
        )
        df[f'gen_{gen_rank}_bus{gen_bus}_ramp_down_normalized'] = (
            negative_ramps.abs().max(axis=1) / (ramp_down_normal + 1e-6)
        )
        
        # Violation counts
        df[f'gen_{gen_rank}_bus{gen_bus}_ramp_up_violations'] = (
            ramps > ramp_up_normal
        ).sum(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_ramp_up_extreme_violations'] = (
            ramps > ramp_up_extreme
        ).sum(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_ramp_down_violations'] = (
            ramps < -ramp_down_normal
        ).sum(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_ramp_down_extreme_violations'] = (
            ramps < -ramp_down_extreme
        ).sum(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_total_ramp_violations'] = (
            df[f'gen_{gen_rank}_bus{gen_bus}_ramp_up_violations'] +
            df[f'gen_{gen_rank}_bus{gen_bus}_ramp_down_violations']
        )
        
        # ===== CATEGORY 3: OPERATIONAL LIMIT FEATURES =====
        # Upper limits
        df[f'gen_{gen_rank}_bus{gen_bus}_hours_near_upper'] = (
            gen_power_df > upper_normal
        ).sum(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_hours_at_extreme_upper'] = (
            gen_power_df > upper_extreme
        ).sum(axis=1)
        
        # Consecutive hours at max (new unique feature)
        df[f'gen_{gen_rank}_bus{gen_bus}_consecutive_hours_at_max'] = rolling_window_max_duration(
            gen_power_df > gen_capacity * 0.95
        )
        
        # Lower limits (when on)
        gen_on = gen_power_df > 0.001
        df[f'gen_{gen_rank}_bus{gen_bus}_hours_near_lower'] = (
            gen_on & (gen_power_df < lower_normal)
        ).sum(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_hours_at_extreme_lower'] = (
            gen_on & (gen_power_df < lower_extreme)
        ).sum(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_hours_below_min_stable'] = (
            gen_on & (gen_power_df < gen_avg * 0.5)
        ).sum(axis=1)
        
        # Severity scores
        df[f'gen_{gen_rank}_bus{gen_bus}_upper_severity'] = (
            gen_power_df.max(axis=1) / gen_capacity
        ).clip(upper=2.0)
        
        min_when_on = gen_power_df[gen_on].min(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_lower_severity'] = (
            1.0 - (min_when_on / gen_avg)
        ).fillna(0).clip(lower=0, upper=1)
        
        # ===== CATEGORY 4: CYCLING & STATE CHANGES =====
        df[f'gen_{gen_rank}_bus{gen_bus}_starts'] = (
            (gen_power_df.shift(1, axis=1) <= 0) & (gen_power_df > 0)
        ).sum(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_stops'] = (
            (gen_power_df.shift(1, axis=1) > 0) & (gen_power_df <= 0)
        ).sum(axis=1)
        df[f'gen_{gen_rank}_bus{gen_bus}_cycles'] = (
            df[f'gen_{gen_rank}_bus{gen_bus}_starts'] + 
            df[f'gen_{gen_rank}_bus{gen_bus}_stops']
        ) / 2
        
        # Duration features
        df[f'gen_{gen_rank}_bus{gen_bus}_longest_on'] = rolling_window_max_duration(gen_on)
        df[f'gen_{gen_rank}_bus{gen_bus}_longest_off'] = rolling_window_max_duration(~gen_on)
        
        # ===== CATEGORY 5: ECONOMIC/MERIT ORDER (UNIQUE) =====
        if gen_bus in merit_order:
            merit_position = merit_order.index(gen_bus)
            
            if merit_position < 5:  # Baseload
                peak_hours = [16, 17, 18, 19, 20]
                peak_output = gen_power_df.iloc[:, peak_hours].mean(axis=1)
                df[f'gen_{gen_rank}_bus{gen_bus}_uneconomic_dispatch'] = (
                    peak_output < gen_avg * 0.7
                ).astype(int)
            else:  # Peaking
                offpeak_hours = [0, 1, 2, 3, 4, 5]
                offpeak_output = gen_power_df.iloc[:, offpeak_hours].mean(axis=1)
                df[f'gen_{gen_rank}_bus{gen_bus}_uneconomic_dispatch'] = (
                    offpeak_output > gen_avg * 1.3
                ).astype(int)
        else:
            df[f'gen_{gen_rank}_bus{gen_bus}_uneconomic_dispatch'] = 0
    
    # ============ GENERATOR FLEET FEATURES ============
    for hour in range(N_HOURS):
        gen_outputs = []
        for gen_bus in generator_buses:
            col = f'ft{hour * FEATURES_PER_HOUR + (gen_bus-1) + 1}'
            if col in df.columns:
                gen_outputs.append(df[col])
        
        if gen_outputs:
            gen_output_df = pd.concat(gen_outputs, axis=1)
            
            df[f'total_generation_h{hour}'] = gen_output_df.sum(axis=1)
            df[f'active_generators_h{hour}'] = (gen_output_df > 0).sum(axis=1)
            df[f'gen_concentration_h{hour}'] = (
                gen_output_df.max(axis=1) / (gen_output_df.sum(axis=1) + 1e-6)
            )
            
            # Generator diversity
            gen_shares = gen_output_df.div(gen_output_df.sum(axis=1) + 1e-6, axis=0)
            entropy_values = []
            for idx in range(len(gen_shares)):
                row_shares = gen_shares.iloc[idx].values
                row_shares = row_shares[row_shares > 0]
                if len(row_shares) > 0:
                    entropy_values.append(stats.entropy(row_shares))
                else:
                    entropy_values.append(0)
            df[f'gen_diversity_h{hour}'] = entropy_values
    
    # ============ MERIT ORDER COMPLIANCE ============
    if len(merit_order) >= 5:
        for hour in range(N_HOURS):
            gen_outputs = []
            gen_buses_in_order = []
            
            for gen_bus in merit_order:
                col = f'ft{hour * FEATURES_PER_HOUR + (gen_bus-1) + 1}'
                if col in df.columns:
                    gen_outputs.append(df[col])
                    gen_buses_in_order.append(gen_bus)
            
            if len(gen_outputs) > 1:
                gen_output_df = pd.concat(gen_outputs, axis=1)
                expected_order = list(range(len(gen_buses_in_order)))
                actual_ranks = gen_output_df.rank(axis=1, ascending=False, method='average')
                
                correlations = []
                inversions = []
                
                for idx in range(len(df)):
                    actual = actual_ranks.iloc[idx].values
                    
                    if len(actual) > 1 and not np.all(actual == actual[0]):
                        corr, _ = stats.spearmanr(expected_order, actual)
                        correlations.append(corr if not np.isnan(corr) else 0)
                    else:
                        correlations.append(0)
                    
                    inv_count = 0
                    for i in range(len(actual)):
                        for j in range(i+1, len(actual)):
                            if expected_order[i] < expected_order[j] and actual[i] > actual[j]:
                                inv_count += 1
                    inversions.append(inv_count)
                
                df[f'merit_order_correlation_h{hour}'] = correlations
                df[f'merit_order_inversions_h{hour}'] = inversions
    
    # ============ ATTACK PATTERN SCORE (for imbalanced datasets) ============
    if feature_store.get('is_imbalanced', False):
        print("Creating combined attack pattern score for imbalanced dataset...")
        
        # Weight different attack types based on their indicators
        attack_scores = pd.DataFrame(index=df.index)
        
        for gen_rank in range(min(len(all_generators), 10)):  # Top 10 generators
            # Type 1: Ramp attacks (use existing features)
            ramp_score = df.get(f'gen_{gen_rank}_bus{all_generators[gen_rank]}_total_ramp_violations', 0) / 10
            
            # Type 2: Upper limit attacks
            upper_score = df.get(f'gen_{gen_rank}_bus{all_generators[gen_rank]}_hours_at_extreme_upper', 0) / 5
            
            # Type 3: Lower limit attacks
            lower_score = df.get(f'gen_{gen_rank}_bus{all_generators[gen_rank]}_hours_below_min_stable', 0) / 5
            
            # Type 4: Economic attacks
            econ_score = df.get(f'gen_{gen_rank}_bus{all_generators[gen_rank]}_uneconomic_dispatch', 0)
            
            # Combined weighted score
            df[f'gen_{gen_rank}_attack_pattern_score'] = (
                0.25 * ramp_score + 0.25 * upper_score + 
                0.25 * lower_score + 0.25 * econ_score
            ).clip(0, 1)
    
    print(f"Created {len([col for col in df.columns if 'gen_' in col])} generator features")
    
    return df, feature_store


def rolling_window_max_duration(boolean_df):
    """Helper function to find maximum consecutive True values"""
    max_durations = []
    for idx in range(len(boolean_df)):
        row = boolean_df.iloc[idx].values
        max_duration = 0
        current_duration = 0
        
        for val in row:
            if val:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        max_durations.append(max_duration)
    
    return pd.Series(max_durations, index=boolean_df.index)





def fix_extreme_features(df, feature_store, is_training=True):
    """Clip extreme outliers that cause numerical instability"""
    
    if is_training:
        clip_bounds = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            if 'deviation' in col or 'distance' in col:
                # Use robust percentiles for clipping
                q01 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                
                # Expand bounds by 2x to be conservative
                lower = q01 - 2 * (q99 - q01)
                upper = q99 + 2 * (q99 - q01)
                
                clip_bounds[col] = (lower, upper)
                df[col] = df[col].clip(lower, upper)
        
        feature_store['clip_bounds'] = clip_bounds
    else:
        # Apply training bounds to test
        clip_bounds = feature_store.get('clip_bounds', {})
        for col, (lower, upper) in clip_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower, upper)
    
    return df, feature_store



    

def diagnose_constant_features(df, feature_store, is_training=True):
    """Identify which features are constant to avoid creating them"""
    
    print("\n=== DIAGNOSING CONSTANT FEATURES ===")
    
    # Group constant features by their creation source
    constant_analysis = {
        'generator_basic': [],
        'generator_ramp': [],
        'generator_limits': [],
        'generator_cycling': [],
        'generator_attack_indicators': [],
        'fleet_features': [],
        'merit_order': [],
        'flow_deviation': [],
        'angle_deviation': [],
        'temporal': [],
        'knn': [],
        'isolation': [],
        'other': []
    }
    
    # Check each feature
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ['attack', 'type_of_attack', 'gen_attacked', 'ssn', 'original_ssn', 'temporal_index']:
            continue
            
        if df[col].var() < 1e-10:  # Constant feature
            # Categorize by name pattern
            if 'gen_' in col and 'bus' in col:
                if any(x in col for x in ['ramp', 'rate']):
                    constant_analysis['generator_ramp'].append(col)
                elif any(x in col for x in ['limit', 'upper', 'lower', 'near', 'extreme']):
                    constant_analysis['generator_limits'].append(col)
                elif any(x in col for x in ['start', 'stop', 'cycle']):
                    constant_analysis['generator_cycling'].append(col)
                elif any(x in col for x in ['excessive', 'sustained', 'forced', 'uneconomic', 'severity']):
                    constant_analysis['generator_attack_indicators'].append(col)
                else:
                    constant_analysis['generator_basic'].append(col)
            elif 'total_generation' in col or 'active_generators' in col or 'gen_concentration' in col or 'gen_diversity' in col:
                constant_analysis['fleet_features'].append(col)
            elif 'merit' in col:
                constant_analysis['merit_order'].append(col)
            elif 'flow_deviation' in col:
                constant_analysis['flow_deviation'].append(col)
            elif 'angle_deviation' in col:
                constant_analysis['angle_deviation'].append(col)
            elif any(x in col for x in ['seasonal', 'temporal', 'lag', 'roll']):
                constant_analysis['temporal'].append(col)
            elif 'knn' in col:
                constant_analysis['knn'].append(col)
            elif 'anomaly' in col or 'attack' in col.lower():
                constant_analysis['isolation'].append(col)
            else:
                constant_analysis['other'].append(col)
    
    # Print analysis
    total_constant = sum(len(v) for v in constant_analysis.values())
    print(f"\nTotal constant features: {total_constant}")
    
    for category, features in constant_analysis.items():
        if features:
            print(f"\n{category.upper()} ({len(features)} constant):")
            # Show first few examples
            for feat in features[:5]:
                value = df[feat].iloc[0] if len(df) > 0 else 'N/A'
                print(f"  - {feat}: constant value = {value}")
            if len(features) > 5:
                print(f"  ... and {len(features)-5} more")
    
    # Identify problematic generators (those producing many constant features)
    problematic_generators = {}
    for feat in constant_analysis['generator_basic'] + constant_analysis['generator_ramp'] + constant_analysis['generator_limits']:
        # Extract bus number from feature name
        import re
        match = re.search(r'bus(\d+)', feat)
        if match:
            bus_num = int(match.group(1))
            if bus_num not in problematic_generators:
                problematic_generators[bus_num] = []
            problematic_generators[bus_num].append(feat)
    
    if problematic_generators:
        print("\n=== PROBLEMATIC GENERATORS (creating constant features) ===")
        sorted_gens = sorted(problematic_generators.items(), key=lambda x: len(x[1]), reverse=True)
        for bus, feats in sorted_gens[:10]:
            if bus in feature_store.get('generator_capacities', {}):
                capacity = feature_store['generator_capacities'][bus]
                avg_gen = feature_store.get('generator_avg_gen', {}).get(bus, 0)
                cap_factor = feature_store.get('generator_capacity_factors', {}).get(bus, 0)
                print(f"  Bus {bus}: {len(feats)} constant features")
                print(f"    Capacity: {capacity:.2f} MW, Avg: {avg_gen:.2f} MW, CF: {cap_factor:.1%}")
                
                # Check if it's a constant output generator
                if cap_factor > 0.99:
                    print(f"    → CONSTANT OUTPUT GENERATOR (100% capacity factor)")
    
    return constant_analysis

def create_bus_baseline_distance_features(df, feature_store, is_training=True):
    """
    Simple baseline distance features to detect phantom generators.
    For each original_ssn, calculate normal bus averages from training normal data,
    then compute distances for all samples.
    """
    N_BUSES = 57
    N_HOURS = 24
    FEATURES_PER_HOUR = 270
    
    print("\n=== Creating Bus Baseline Distance Features ===")
    
    if is_training:
        # Build baselines from normal training data only
        normal_df = df[df['attack'] == 0].copy()
        
        bus_baselines = {}
        
        for ssn in normal_df['original_ssn'].unique():
            ssn_normal = normal_df[normal_df['original_ssn'] == ssn]
            bus_baselines[ssn] = {}
            
            for hour in range(N_HOURS):
                # Get the first 14 features (bus powers) for this hour
                bus_means = []
                for bus_idx in range(N_BUSES):
                    col = f'ft{hour * FEATURES_PER_HOUR + bus_idx + 1}'
                    if col in ssn_normal.columns:
                        bus_means.append(ssn_normal[col].mean())
                    else:
                        bus_means.append(0)
                
                bus_baselines[ssn][hour] = np.array(bus_means)
        
        feature_store['bus_baselines'] = bus_baselines
        print(f"Created baselines for {len(bus_baselines)} SSNs")
    
    # Calculate distances for all data
    bus_baselines = feature_store.get('bus_baselines', {})
    
    # Initialize features
    for hour in range(N_HOURS):
        df[f'bus_dist_h{hour}'] = 0.0
    
    df['bus_dist_mean'] = 0.0
    df['bus_dist_max'] = 0.0
    df['phantom_score'] = 0.0
    
    print("Calculating distances from baselines...")
    
    for ssn in df['original_ssn'].unique():
        if ssn not in bus_baselines:
            continue
            
        ssn_mask = df['original_ssn'] == ssn
        ssn_data = df[ssn_mask]
        
        hour_distances = []
        
        for hour in range(N_HOURS):
            baseline = bus_baselines[ssn][hour]
            
            # Get actual bus values
            actual_values = []
            for bus_idx in range(N_BUSES):
                col = f'ft{hour * FEATURES_PER_HOUR + bus_idx + 1}'
                if col in ssn_data.columns:
                    actual_values.append(ssn_data[col].values)
                else:
                    actual_values.append(np.zeros(len(ssn_data)))
            
            actual_array = np.array(actual_values).T  # Shape: (n_samples, n_buses)
            
            # Calculate Euclidean distance for each sample
            distances = np.sqrt(np.sum((actual_array - baseline)**2, axis=1))
            
            df.loc[ssn_mask, f'bus_dist_h{hour}'] = distances
            hour_distances.append(distances)
            
            # Phantom score: extra penalty for non-zero values where baseline is zero
            phantom_penalty = np.zeros(len(ssn_data))
            for bus_idx in range(N_BUSES):
                if baseline[bus_idx] < 0.01:  # Bus normally doesn't generate
                    # Penalize if this bus is now generating
                    phantom_penalty += (actual_array[:, bus_idx] > 0.1).astype(float) * actual_array[:, bus_idx]
            
            df.loc[ssn_mask, 'phantom_score'] += phantom_penalty
        
        # Aggregate features
        if hour_distances:
            hour_distances = np.array(hour_distances).T
            df.loc[ssn_mask, 'bus_dist_mean'] = hour_distances.mean(axis=1)
            df.loc[ssn_mask, 'bus_dist_max'] = hour_distances.max(axis=1)
    
    # Create simple alert flags
    df['high_bus_distance'] = (df['bus_dist_mean'] > df['bus_dist_mean'].quantile(0.95)).astype(int)
    df['phantom_detected'] = (df['phantom_score'] > 0).astype(int)
    
    if is_training:
        print(f"\nDistance Statistics:")
        print(f"Normal: mean={df[df['attack']==0]['bus_dist_mean'].mean():.3f}")
        print(f"Attack: mean={df[df['attack']==1]['bus_dist_mean'].mean():.3f}")
        
        print(f"\nPhantom Detection:")
        print(f"Normal with phantom: {df[df['attack']==0]['phantom_detected'].sum()}")
        print(f"Attacks with phantom: {df[df['attack']==1]['phantom_detected'].sum()}")
    
    return df, feature_store


def engineer_advanced_features(df, is_training=False, feature_store=None):
    
    
    original_shape = df.shape
    
    # Initialize feature store
    if feature_store is None:
        feature_store = {}
    
    # Detect imbalance
    if is_training:
        attack_ratio = df["attack"].mean()
        feature_store['attack_ratio'] = attack_ratio
        feature_store["is_imbalanced"] = attack_ratio < 0.4
        
        print(f"\n=== Imbalance Detection ===")
        print(f"Attack ratio in training: {attack_ratio:.2%}")
        print(f"Dataset classification: {'Imbalanced' if feature_store['is_imbalanced'] else 'Balanced'}")
    
    # Sort by temporal order
    df = df.sort_values(['original_ssn', 'temporal_index']).reset_index(drop=True)
    
    if not is_training:
        print(f"Test data sessions: {sorted(df['ssn'].unique())}")
        print(f"Test data temporal range: {df['temporal_index'].min()} to {df['temporal_index'].max()}")
    
    # Step 1: Identify OPF columns and keep only the informative ones, never used, but we kept them for future reference
    opf_columns = [col for col in df.columns if col.startswith('ft')]
    opf_columns = [col for col in opf_columns if col in df.columns]
    

   
    # Step 2: Identify generators from normal data
    if is_training:
        feature_store = identify_generator_buses(df, feature_store, verbose=True)
    
    # Step 3: Calculate data-driven limits (not fixed percentages)
    if is_training:
        feature_store = calculate_normal_operational_limits(df, feature_store, is_training)
        feature_store = calculate_bidirectional_ramp_limits(df, feature_store, is_training)
    
    # Step 4: Calculate normal baselines for flows and angles
    df, feature_store = calculate_normal_flow_and_angle_baselines(df, feature_store, is_training)
    
    # Step 5: NEW - Create bus baseline distance features for phantom generator detection
    df, feature_store = create_bus_baseline_distance_features(df, feature_store, is_training)

   
    # Step 6: Create features using data-driven limits
    df, feature_store = engineer_generator_focused_features(df, is_training=True, feature_store=feature_store)
   
    # Step 7: Create improved k-NN models with scaling and thresholds 
    df, feature_store = create_improved_knn_models(df, feature_store, is_training)

    
    # Drop original OPF features
    print(f"Dropping {len(opf_columns)} OPF features after engineering...")
    df = df.drop(columns=opf_columns, errors='ignore')
    print(f"Dropped original OPF features (information preserved in engineered features)")
    
    # if is_training:
    #     # Diagnose constant features to improve future runs
    #     constant_analysis = diagnose_constant_features(df, feature_store, is_training)
    #     feature_store['constant_analysis'] = constant_analysis
    

    
    # Step 8: Fix extreme outliers
    df, feature_store = fix_extreme_features(df, feature_store, is_training)
    
    
 
    
    # ============ CLEANUP ============
    # Fill NaNs
    df.fillna(0, inplace=True)

    
    
    # Final diagnostics
    new_shape = df.shape
    features_added = new_shape[1] - original_shape[1] + len(opf_columns)
    
    
    print(f"\n✅ Feature engineering completed!")
    print(f"Original shape: {original_shape}")
    print(f"Final shape: {new_shape}")
    print(f"Net features added: {features_added}")
    
    
    return df, feature_store



def validate_temporal_features(train_data, test_data):
    """Validate that features don't leak future information"""
    print("\n=== Temporal Feature Validation ===")
    
    # Only process numeric columns
    numeric_cols = train_data.select_dtypes(include=np.number).columns
    
    suspicious_temporal = []
    
    for col in numeric_cols:
        try:
            train_min = train_data[col].min()
            train_max = train_data[col].max()
            test_min = test_data[col].min()
            test_max = test_data[col].max()
            
            # Calculate safe buffer zone
            buffer_low = 0.1 * abs(train_min) if train_min != 0 else 0.1
            buffer_high = 0.1 * abs(train_max) if train_max != 0 else 0.1
            
            if test_min < train_min - buffer_low or test_max > train_max + buffer_high:
                suspicious_temporal.append({
                    'feature': col,
                    'train_range': (train_min, train_max),
                    'test_range': (test_min, test_max),
                    'expansion': max(abs(test_min - train_min), abs(test_max - train_max))
                })
        except TypeError:
            # Skip non-numeric columns that slip through
            continue
            
    if suspicious_temporal:
        print(f"\nFound {len(suspicious_temporal)} features with suspicious test set expansion:")
        for feat in sorted(suspicious_temporal, key=lambda x: x['expansion'], reverse=True)[:5]:
            print(f"\n{feat['feature']}:")
            print(f"  Train range: [{feat['train_range'][0]:.4f}, {feat['train_range'][1]:.4f}]")
            print(f"  Test range: [{feat['test_range'][0]:.4f}, {feat['test_range'][1]:.4f}]")
    
    return suspicious_temporal
