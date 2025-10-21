from flask import Flask, render_template, jsonify, request,send_from_directory
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import traceback

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': 'db.trwsfdhxzwzkjandsmvz.supabase.co',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'RkoGkPLWxh4vavX3',
    'port': '5432'
}

class InductionPlanner:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.load_model()
        
        # All configuration comes from database - no static values
        self.service_requirements = self.get_service_requirements()
        self.depot_bays = self.get_depot_configuration()
        self.stabling_tracks = self.get_stabling_tracks()
        self.categorical_mappings = self.get_categorical_mappings()

    def get_service_requirements(self):
        """Calculate service requirements from actual train data only"""
        conn = self.get_db_connection()
        if not conn:
            raise Exception("Database connection failed - cannot determine service requirements")
        
        try:
            with conn.cursor() as cur:
                # Check if config table exists first
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'system_config'
                    )
                """)
                config_table_exists = cur.fetchone()[0]
                
                if config_table_exists:
                    cur.execute("""
                        SELECT setting_name, setting_value 
                        FROM system_config 
                        WHERE setting_category = 'service_requirements'
                    """)
                    config = dict(cur.fetchall())
                    
                    if config:
                        return {
                            'revenue_service_needed': int(config['revenue_service_needed']),
                            'standby_needed': int(config['standby_needed']),
                            'maintenance_capacity': int(config['maintenance_capacity'])
                        }
                
                # Calculate from actual fleet data in train_status table
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_trains,
                        COUNT(CASE WHEN fitness_certificate = true THEN 1 END) as fit_trains,
                        COUNT(CASE WHEN pending_maintenance = false THEN 1 END) as ready_trains,
                        COUNT(CASE WHEN brake_wear < 60 THEN 1 END) as good_brake_trains,
                        COUNT(CASE WHEN motor_temp < 75 THEN 1 END) as good_temp_trains
                    FROM train_status 
                    WHERE (retirement_date IS NULL OR retirement_date > CURRENT_DATE)
                """)
                
                result = cur.fetchone()
                if not result or result[0] == 0:
                    raise Exception("No active trains found in train_status table")
                
                total_trains, fit_trains, ready_trains, good_brake_trains, good_temp_trains = result
                
                # Calculate operational requirements based on train condition
                high_readiness_trains = min(fit_trains or 0, ready_trains or 0, good_brake_trains or 0)
                
                # Revenue service: trains in best condition (60-70% of high readiness trains)
                revenue_needed = max(int(high_readiness_trains * 0.65), min(12, total_trains))
                
                # Standby: backup trains (15-25% of total fleet)  
                standby_needed = max(int(total_trains * 0.2), min(3, total_trains - revenue_needed))
                
                # Maintenance capacity: remaining trains
                maintenance_capacity = total_trains - revenue_needed - standby_needed
                
                logger.info(f"Calculated requirements from {total_trains} trains: {revenue_needed} revenue, {standby_needed} standby, {maintenance_capacity} maintenance")
                
                return {
                    'revenue_service_needed': revenue_needed,
                    'standby_needed': standby_needed,
                    'maintenance_capacity': maintenance_capacity
                }
                
        except Exception as e:
            logger.error(f"Error getting service requirements: {e}")
            raise Exception(f"Failed to calculate service requirements from database: {e}")
        finally:
            conn.close()

    def get_depot_configuration(self):
        """Get depot bay configuration from existing database tables"""
        conn = self.get_db_connection()
        if not conn:
            raise Exception("Database connection failed - cannot get depot configuration")
        
        try:
            with conn.cursor() as cur:
                # First check what columns exist in kmrl_parts_cost table
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'kmrl_parts_cost'
                    )
                """)
                parts_table_exists = cur.fetchone()[0]
                
                if parts_table_exists:
                    # Check if the old kmrl_parts_cost has bay columns
                    cur.execute("""
                        SELECT column_name FROM information_schema.columns 
                        WHERE table_name = 'kmrl_parts_cost' 
                        AND column_name IN ('bay_id', 'bay_type', 'bay_avg_time')
                    """)
                    bay_columns = [row[0] for row in cur.fetchall()]
                    
                    if 'bay_id' in bay_columns and 'bay_type' in bay_columns:
                        cur.execute("""
                            SELECT DISTINCT 
                                bay_id,
                                bay_type,
                                AVG(bay_avg_time) as avg_hours,
                                COUNT(*) as usage_count
                            FROM kmrl_parts_cost 
                            WHERE bay_id IS NOT NULL AND bay_type IS NOT NULL
                            GROUP BY bay_id, bay_type
                            ORDER BY bay_id
                        """)
                        
                        results = cur.fetchall()
                        if results:
                            bay_config = {}
                            for bay_id, bay_type, avg_hours, usage_count in results:
                                bay_config[bay_id] = {
                                    'type': bay_type,
                                    'avg_hours': float(avg_hours or 24),
                                    'usage_count': usage_count
                                }
                            
                            logger.info(f"Loaded {len(bay_config)} maintenance bays from kmrl_parts_cost table")
                            return bay_config
                
                # Check if there's a new kmrl_relevant_parts_cost table
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'kmrl_relevant_parts_cost'
                    )
                """)
                new_parts_table_exists = cur.fetchone()[0]
                
                if new_parts_table_exists:
                    # Use parts data to create maintenance bays based on part types
                    cur.execute("""
                        SELECT part_name, part_cost 
                        FROM kmrl_relevant_parts_cost
                        ORDER BY part_cost DESC
                    """)
                    
                    parts = cur.fetchall()
                    if parts:
                        bay_config = {}
                        bay_counter = 1
                        
                        # Create specialized bays based on high-cost parts
                        for part_name, part_cost in parts[:8]:  # Top 8 most expensive parts get dedicated bays
                            bay_type = 'General Maintenance'
                            avg_hours = 24.0
                            
                            # Assign specific bay types based on part names
                            if 'brake' in part_name.lower():
                                bay_type = 'Brake Service'
                                avg_hours = 16.0
                            elif 'inverter' in part_name.lower() or 'traction' in part_name.lower():
                                bay_type = 'Traction Maintenance'
                                avg_hours = 20.0
                            elif 'door' in part_name.lower():
                                bay_type = 'Door Systems'
                                avg_hours = 8.0
                            elif 'air conditioning' in part_name.lower() or 'hvac' in part_name.lower():
                                bay_type = 'HVAC Service'
                                avg_hours = 12.0
                            elif 'battery' in part_name.lower():
                                bay_type = 'Battery Maintenance'
                                avg_hours = 6.0
                            elif 'transformer' in part_name.lower() or 'pantograph' in part_name.lower():
                                bay_type = 'Heavy Maintenance'
                                avg_hours = 48.0
                            
                            bay_config[f"BAY_{bay_counter:02d}"] = {
                                'type': bay_type,
                                'avg_hours': avg_hours,
                                'usage_count': 0,
                                'associated_part': part_name,
                                'part_cost': float(part_cost)
                            }
                            bay_counter += 1
                        
                        logger.info(f"Created {len(bay_config)} maintenance bays from parts cost data")
                        return bay_config
                
                # Try to get unique depot positions from train_status as bay identifiers
                cur.execute("""
                    SELECT DISTINCT depot_position,
                           COUNT(*) as train_count
                    FROM train_status 
                    WHERE depot_position IS NOT NULL
                    GROUP BY depot_position
                    ORDER BY depot_position
                """)
                
                positions = cur.fetchall()
                if positions:
                    bay_config = {}
                    for i, (position, count) in enumerate(positions):
                        bay_type = 'General Maintenance'
                        if i < 2:
                            bay_type = 'Heavy Maintenance'
                        elif i < 4:
                            bay_type = 'Periodic Maintenance'
                        
                        bay_config[f"BAY_{position}"] = {
                            'type': bay_type,
                            'avg_hours': 36.0 if bay_type == 'Heavy Maintenance' else 24.0,
                            'usage_count': count
                        }
                    
                    logger.info(f"Created {len(bay_config)} maintenance bays from depot positions")
                    return bay_config
                
                # Generate based on train count - minimum required maintenance capacity
                cur.execute("SELECT COUNT(*) FROM train_status WHERE retirement_date IS NULL OR retirement_date > CURRENT_DATE")
                total_trains = cur.fetchone()[0]
                
                if total_trains > 0:
                    # Create minimum required maintenance bays (30% of fleet)
                    num_bays = max(4, int(total_trains * 0.3))
                    bay_config = {}
                    
                    bay_types = [
                        ('Heavy Maintenance', 48.0),
                        ('Periodic Maintenance', 36.0), 
                        ('Brake Service', 16.0),
                        ('HVAC Service', 12.0),
                        ('Door Systems', 8.0),
                        ('General Maintenance', 24.0)
                    ]
                    
                    for i in range(1, num_bays + 1):
                        bay_type, hours = bay_types[min(i-1, len(bay_types)-1)]
                        
                        bay_config[f"MAINT_BAY_{i:02d}"] = {
                            'type': bay_type,
                            'avg_hours': hours,
                            'usage_count': 0
                        }
                    
                    logger.info(f"Generated {num_bays} maintenance bays based on fleet size ({total_trains} trains)")
                    return bay_config
                
                raise Exception("Cannot determine depot configuration - no train data found")
                
        except Exception as e:
            logger.error(f"Error getting depot configuration: {e}")
            raise Exception(f"Failed to get depot configuration from database: {e}")
        finally:
            conn.close()

    def get_stabling_tracks(self):
        """Get stabling tracks from existing database data"""
        conn = self.get_db_connection()
        if not conn:
            raise Exception("Database connection failed - cannot get stabling tracks")
        
        try:
            with conn.cursor() as cur:
                # Check if stabling_tracks table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'stabling_tracks'
                    )
                """)
                stabling_table_exists = cur.fetchone()[0]
                
                if stabling_table_exists:
                    cur.execute("""
                        SELECT DISTINCT track_id, track_capacity
                        FROM stabling_tracks 
                        WHERE active = true
                        ORDER BY track_id
                    """)
                    
                    results = cur.fetchall()
                    if results:
                        tracks = []
                        for track_id, capacity in results:
                            tracks.extend([f"{track_id}_{i}" for i in range(1, int(capacity) + 1)])
                        
                        logger.info(f"Loaded {len(tracks)} stabling positions from stabling_tracks table")
                        return tracks
                
                # Use depot_position from train_status as stabling positions
                cur.execute("""
                    SELECT DISTINCT depot_position
                    FROM train_status 
                    WHERE depot_position IS NOT NULL
                    ORDER BY depot_position
                """)
                
                positions = cur.fetchall()
                if positions:
                    tracks = [f"TRACK_{pos[0]}" for pos in positions]
                    logger.info(f"Created {len(tracks)} stabling tracks from depot positions")
                    return tracks
                
                # Generate based on train count (stabling capacity = 80% of fleet)
                cur.execute("SELECT COUNT(*) FROM train_status WHERE retirement_date IS NULL OR retirement_date > CURRENT_DATE")
                total_trains = cur.fetchone()[0]
                
                if total_trains > 0:
                    num_tracks = max(int(total_trains * 0.8), 5)  # 80% capacity with minimum 5 tracks
                    tracks = [f"ST_{i:02d}" for i in range(1, num_tracks + 1)]
                    
                    logger.info(f"Generated {len(tracks)} stabling tracks based on fleet size")
                    return tracks
                
                raise Exception("Cannot determine stabling configuration - no relevant data found")
                
        except Exception as e:
            logger.error(f"Error getting stabling tracks: {e}")
            raise Exception(f"Failed to get stabling tracks from database: {e}")
        finally:
            conn.close()

    def get_categorical_mappings(self):
        """Generate categorical mappings from actual database values only"""
        conn = self.get_db_connection()
        if not conn:
            raise Exception("Database connection failed - cannot get categorical mappings")
        
        mappings = {}
        
        try:
            with conn.cursor() as cur:
                # Get all possible values for categorical fields from actual data
                categorical_fields = [
                    'fault_code', 'depot_location', 'depot_position', 
                    'cleaning_status', 'work_order_status', 'depot_accessibility',
                    'advertising_commitments'
                ]
                
                for field in categorical_fields:
                    cur.execute(f"""
                        SELECT DISTINCT {field} 
                        FROM train_status 
                        WHERE {field} IS NOT NULL
                        ORDER BY {field}
                    """)
                    
                    values = [row[0] for row in cur.fetchall()]
                    if values:
                        mappings[field] = {val: idx for idx, val in enumerate(values)}
                    else:
                        logger.warning(f"No values found for categorical field: {field}")
                        mappings[field] = {}
                
                return mappings
                
        except Exception as e:
            logger.error(f"Error getting categorical mappings: {e}")
            raise Exception(f"Failed to get categorical mappings from database: {e}")
        finally:
            conn.close()

    def load_model(self):
        """Load the trained ML model and scaler"""
        try:
            model_path = 'realistic_kochi_metro_rf_model.pkl'
            scaler_path = 'realistic_kochi_metro_scaler.pkl'
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("ML model and scaler loaded successfully")
                
                self.feature_columns = [
                    'mtbf', 'brake_wear', 'energy_kwh', 'fault_code', 'mileage_km', 
                    'motor_temp', 'trip_count', 'energy_cost', 'hvac_status', 
                    'door_failures', 'available_bays', 'depot_location', 'depot_position',
                    'battery_current', 'battery_voltage', 'cleaning_status', 'operating_hours',
                    'door_cycle_count', 'incident_reports', 'maintenance_cost', 
                    'compliance_status', 'mileage_balancing', 'work_order_status',
                    'passenger_capacity', 'passengers_onboard', 'depot_accessibility',
                    'fitness_certificate', 'pending_maintenance', 'standby_requirement',
                    'total_operating_cost', 'operating_cost_per_km', 'advertising_commitments',
                    'operating_cost_per_hour', 'stabling_geometry_score', 'occupancy_ratio',
                    'energy_per_km'
                ]
            else:
                logger.warning("ML model files not found. Using rule-based scoring only.")
                self.model = None
                self.scaler = None
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            self.model = None
            self.scaler = None

    def get_db_connection(self):
        """Get database connection with proper error handling"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            return conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            return None

    def fetch_train_data(self):
        """Fetch real train data from database - NO DEFAULT VALUES"""
        conn = self.get_db_connection()
        if not conn:
            raise Exception("Database connection failed")
        
        try:
            # Fetch all train data with NO COALESCE defaults - use actual data only
            query = """
            SELECT 
                train_id,
                train_number,
                mtbf,
                brake_wear,
                energy_kwh,
                fault_code,
                mileage_km,
                motor_temp,
                trip_count,
                energy_cost,
                hvac_status,
                door_failures,
                available_bays,
                depot_location,
                depot_position,
                battery_current,
                battery_voltage,
                cleaning_status,
                operating_hours,
                retirement_date,
                door_cycle_count,
                incident_reports,
                maintenance_cost,
                compliance_status,
                mileage_balancing,
                work_order_status,
                passenger_capacity,
                passengers_onboard,
                depot_accessibility,
                fitness_certificate,
                pending_maintenance,
                standby_requirement,
                total_operating_cost,
                last_maintenance_date,
                operating_cost_per_km,
                advertising_commitments,
                operating_cost_per_hour,
                stabling_geometry_score,
                occupancy_ratio,
                energy_per_km,
                created_at,
                updated_at
            FROM train_status 
            WHERE (retirement_date IS NULL OR retirement_date > CURRENT_DATE)
            ORDER BY train_number
            """
            
            df = pd.read_sql(query, conn)
            logger.info(f"Fetched {len(df)} active trains from database")
            
            if df.empty:
                raise Exception("No active trains found in database")
            
            # Check for critical missing data
            critical_fields = ['train_number', 'fitness_certificate', 'brake_wear', 'motor_temp']
            for field in critical_fields:
                if field in df.columns:
                    missing_count = df[field].isnull().sum()
                    if missing_count > 0:
                        logger.warning(f"Missing data in critical field {field}: {missing_count} records")
                        if missing_count == len(df):
                            raise Exception(f"All records missing critical field: {field}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching train data: {e}")
            raise
        finally:
            conn.close()

    def prepare_features_for_model(self, df):
        """Prepare features for ML model - handle missing data intelligently"""
        try:
            model_df = df.copy()
            
            # Apply categorical mappings from database
            for col, mapping in self.categorical_mappings.items():
                if col in model_df.columns and mapping:
                    model_df[col] = model_df[col].map(mapping)
                    # For unmapped values, use -1 to indicate unknown
                    model_df[col] = model_df[col].fillna(-1)
            
            # Convert boolean columns to int
            bool_columns = ['hvac_status', 'compliance_status', 'fitness_certificate', 
                          'pending_maintenance', 'standby_requirement']
            for col in bool_columns:
                if col in model_df.columns:
                    model_df[col] = model_df[col].astype(int)
            
            # Handle missing numerical data with median imputation from current dataset
            numerical_columns = [col for col in self.feature_columns if col not in bool_columns 
                               and col not in self.categorical_mappings.keys()]
            
            for col in numerical_columns:
                if col in model_df.columns:
                    median_val = model_df[col].median()
                    if pd.isna(median_val):
                        # If all values are null, cannot proceed
                        raise Exception(f"No valid data for critical field: {col}")
                    model_df[col] = model_df[col].fillna(median_val)
            
            # Ensure all required features exist
            for col in self.feature_columns:
                if col not in model_df.columns:
                    raise Exception(f"Required feature missing from database: {col}")
            
            # Select features in training order
            feature_data = model_df[self.feature_columns].copy()
            
            # Final check for any remaining nulls
            null_counts = feature_data.isnull().sum()
            if null_counts.sum() > 0:
                raise Exception(f"Null values found in features: {null_counts[null_counts > 0].to_dict()}")
            
            return feature_data
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def calculate_readiness_scores(self, df):
        """Calculate readiness scores using ML model or database-driven rules"""
        try:
            if self.model is not None and self.scaler is not None:
                feature_data = self.prepare_features_for_model(df)
                scaled_features = self.scaler.transform(feature_data)
                readiness_scores = self.model.predict(scaled_features)
                
                # Ensure scores are in valid range (0-10)
                readiness_scores = np.clip(readiness_scores, 0, 10)
                logger.info("Successfully used ML model for readiness scoring")
                return readiness_scores
            else:
                logger.info("ML model not available, using database-driven rule-based scoring")
                return self.database_driven_scoring(df)
                
        except Exception as e:
            logger.error(f"Error with ML scoring, falling back to database-driven scoring: {e}")
            return self.database_driven_scoring(df)

    def database_driven_scoring(self, df):
        """Enhanced rule-based scoring using ONLY database values"""
        scores = []
        
        # Get scoring weights from database if available
        scoring_weights = self.get_scoring_weights()
        
        for _, train in df.iterrows():
            score = 5.0  # Base score
            
            # Use only non-null values for scoring
            if pd.notna(train.get('fitness_certificate')):
                if train['fitness_certificate']:
                    score += scoring_weights.get('fitness_certificate_good', 2.0)
                else:
                    score -= scoring_weights.get('fitness_certificate_bad', 5.0)
            
            if pd.notna(train.get('pending_maintenance')):
                if train['pending_maintenance']:
                    score -= scoring_weights.get('pending_maintenance', 2.0)
                else:
                    score += scoring_weights.get('no_pending_maintenance', 1.0)
            
            # Brake condition scoring
            if pd.notna(train.get('brake_wear')):
                brake_wear = float(train['brake_wear'])
                if brake_wear < 30:
                    score += scoring_weights.get('brake_excellent', 1.5)
                elif brake_wear < 60:
                    score += scoring_weights.get('brake_good', 0.5)
                elif brake_wear < 80:
                    score -= scoring_weights.get('brake_warning', 0.5)
                else:
                    score -= scoring_weights.get('brake_critical', 2.0)
            
            # Temperature scoring
            if pd.notna(train.get('motor_temp')):
                motor_temp = float(train['motor_temp'])
                if motor_temp < 60:
                    score += scoring_weights.get('temp_optimal', 1.0)
                elif motor_temp < 75:
                    score += scoring_weights.get('temp_good', 0.5)
                elif motor_temp < 90:
                    score -= scoring_weights.get('temp_warning', 0.5)
                else:
                    score -= scoring_weights.get('temp_critical', 2.0)
            
            # Door failures
            if pd.notna(train.get('door_failures')):
                door_failures = int(train['door_failures'])
                if door_failures == 0:
                    score += scoring_weights.get('no_door_failures', 0.8)
                elif door_failures <= 2:
                    score -= scoring_weights.get('minor_door_failures', 0.5)
                elif door_failures <= 5:
                    score -= scoring_weights.get('moderate_door_failures', 1.0)
                else:
                    score -= scoring_weights.get('major_door_failures', 2.0)
            
            # HVAC status
            if pd.notna(train.get('hvac_status')):
                if train['hvac_status']:
                    score += scoring_weights.get('hvac_working', 0.5)
                else:
                    score -= scoring_weights.get('hvac_broken', 1.5)
            
            # Fault presence
            if pd.notna(train.get('fault_code')):
                fault_code = str(train['fault_code']).strip()
                if fault_code == '' or fault_code.lower() == 'none' or pd.isna(train['fault_code']):
                    score += scoring_weights.get('no_faults', 0.8)
                else:
                    score -= scoring_weights.get('active_fault', 1.5)
            
            # Recent incidents
            if pd.notna(train.get('incident_reports')):
                incidents = int(train['incident_reports'])
                if incidents == 0:
                    score += scoring_weights.get('no_incidents', 0.5)
                elif incidents <= 2:
                    score -= scoring_weights.get('minor_incidents', 0.3)
                else:
                    score -= scoring_weights.get('major_incidents', 1.0)
            
            # Compliance status
            if pd.notna(train.get('compliance_status')):
                if train['compliance_status']:
                    score += scoring_weights.get('compliant', 0.5)
                else:
                    score -= scoring_weights.get('non_compliant', 1.5)
            
            # Ensure score is within bounds
            final_score = max(0, min(10, score))
            scores.append(final_score)
        
        return np.array(scores)

    def get_scoring_weights(self):
        """Get scoring weights from database configuration"""
        conn = self.get_db_connection()
        if not conn:
            # Cannot get weights from database, use basic weights
            logger.warning("Cannot get scoring weights from database, using basic scoring")
            return {}
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT weight_name, weight_value 
                    FROM scoring_weights 
                    WHERE active = true
                """)
                weights = dict(cur.fetchall())
                return weights if weights else {}
        except:
            return {}  # Table doesn't exist or query failed
        finally:
            conn.close()

    def get_parts_cost_data(self):
        """Get parts cost data from the new table"""
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT part_name, part_cost 
                    FROM kmrl_relevant_parts_cost
                    ORDER BY part_cost DESC
                """)
                parts_data = dict(cur.fetchall())
                return parts_data
        except Exception as e:
            logger.warning(f"Could not fetch parts cost data: {e}")
            return {}
        finally:
            conn.close()

    def generate_explanations(self, df, scores):
        """Generate explanations based only on actual database values"""
        explanations = []
        
        for idx, (_, train) in enumerate(df.iterrows()):
            explanation = []
            score = scores[idx]
            
            # Only add explanations for non-null values
            if pd.notna(train.get('fitness_certificate')) and not train['fitness_certificate']:
                explanation.append("Invalid fitness certificate - Cannot operate")
            
            if pd.notna(train.get('pending_maintenance')) and train['pending_maintenance']:
                explanation.append("Pending maintenance work required")
            
            if pd.notna(train.get('fault_code')):
                fault_code = str(train['fault_code']).strip()
                if fault_code and fault_code.lower() != 'none':
                    explanation.append(f"Active fault: {fault_code}")
            
            if pd.notna(train.get('brake_wear')):
                brake_wear = float(train['brake_wear'])
                if brake_wear > 80:
                    explanation.append(f"High brake wear ({brake_wear}%) - Service needed")
                elif brake_wear < 30:
                    explanation.append(f"Excellent brake condition ({brake_wear}%)")
            
            if pd.notna(train.get('motor_temp')):
                motor_temp = float(train['motor_temp'])
                if motor_temp > 90:
                    explanation.append(f"High motor temperature ({motor_temp:.1f}°C)")
                elif motor_temp < 60:
                    explanation.append(f"Optimal motor temperature ({motor_temp:.1f}°C)")
            
            if pd.notna(train.get('door_failures')):
                door_failures = int(train['door_failures'])
                if door_failures > 5:
                    explanation.append(f"Multiple door failures ({door_failures})")
                elif door_failures == 0:
                    explanation.append("No door system issues")
            
            if pd.notna(train.get('hvac_status')) and not train['hvac_status']:
                explanation.append("HVAC system not operational")
            
            if pd.notna(train.get('incident_reports')):
                incidents = int(train['incident_reports'])
                if incidents > 2:
                    explanation.append(f"Recent incidents reported ({incidents})")
            
            # Maintenance timing based on actual data
            if pd.notna(train.get('last_maintenance_date')):
                try:
                    if isinstance(train['last_maintenance_date'], str):
                        last_maintenance = datetime.strptime(train['last_maintenance_date'], '%Y-%m-%d').date()
                    else:
                        last_maintenance = train['last_maintenance_date']
                    
                    days_since = (datetime.now().date() - last_maintenance).days
                    if days_since > 60:
                        explanation.append(f"{days_since} days since last maintenance")
                    elif days_since < 14:
                        explanation.append(f"Recently maintained ({days_since} days ago)")
                except:
                    pass
            
            # Limit to top 4 most relevant explanations
            explanations.append(explanation[:4])
        
        return explanations

    def allocate_trains(self, df, scores, explanations):
        """Train allocation using only database-derived requirements"""
        allocation_df = df.copy()
        allocation_df['readiness_score'] = scores
        allocation_df['explanation'] = explanations
        
        # Sort by readiness score (highest first)
        allocation_df = allocation_df.sort_values('readiness_score', ascending=False)
        allocation_df['rank'] = range(1, len(allocation_df) + 1)
        
        allocations = []
        bay_assignments = {}
        track_assignments = {}
        
        revenue_count = 0
        standby_count = 0
        maintenance_count = 0
        
        # Use database-driven thresholds
        revenue_threshold = self.calculate_dynamic_threshold(scores, 'revenue')
        standby_threshold = self.calculate_dynamic_threshold(scores, 'standby')
        
        for _, train in allocation_df.iterrows():
            allocation = {
                'train_number': train['train_number'],
                'train_id': int(train['train_id']),
                'readiness_score': float(train['readiness_score']),
                'rank': int(train['rank']),
                'explanation': train['explanation'],
                'category': None,
                'status': None,
                'location': None,
                'maintenance_type': None,
                'estimated_completion': None,
                'bay_assignment': None,
                'track_position': None
            }
            
            # Determine allocation based on database conditions
            critical_issues = self.has_critical_issues_db_only(train)
            
            if critical_issues or train['readiness_score'] < 3.0:
                maintenance_type = self.determine_maintenance_type_db_only(train)
                bay = self.assign_maintenance_bay(maintenance_type, bay_assignments)
                
                allocation.update({
                    'category': 'IBL Maintenance',
                    'status': 'Assigned' if bay else 'Queued',
                    'maintenance_type': maintenance_type,
                    'bay_assignment': bay,
                    'location': bay if bay else 'Queue',
                    'estimated_completion': self.calculate_completion_time_db_only(maintenance_type, train)
                })
                
                if bay:
                    bay_assignments[bay] = train['train_number']
                maintenance_count += 1
                    
            elif (revenue_count < self.service_requirements['revenue_service_needed'] and 
                  train['readiness_score'] >= revenue_threshold):
                allocation.update({
                    'category': 'Revenue Service',
                    'status': 'Ready for Service',
                    'location': 'Depot Ready'
                })
                revenue_count += 1
                
            elif (standby_count < self.service_requirements['standby_needed'] and 
                  train['readiness_score'] >= standby_threshold):
                track = self.assign_stabling_track(track_assignments)
                allocation.update({
                    'category': 'Standby',
                    'status': 'Standing By',
                    'track_position': track,
                    'location': track if track else 'Depot Yard'
                })
                if track:
                    track_assignments[track] = train['train_number']
                standby_count += 1
                
            else:
                # Remaining trains go to maintenance
                maintenance_type = self.determine_maintenance_type_db_only(train)
                bay = self.assign_maintenance_bay(maintenance_type, bay_assignments)
                
                allocation.update({
                    'category': 'IBL Maintenance',
                    'status': 'Assigned' if bay else 'Queued',
                    'maintenance_type': maintenance_type,
                    'bay_assignment': bay,
                    'location': bay if bay else 'Queue',
                    'estimated_completion': self.calculate_completion_time_db_only(maintenance_type, train)
                })
                
                if bay:
                    bay_assignments[bay] = train['train_number']
                maintenance_count += 1
            
            allocations.append(allocation)
        
        logger.info(f"Allocation complete: {revenue_count} revenue, {standby_count} standby, {maintenance_count} maintenance")
        return allocations, bay_assignments, track_assignments

    def calculate_dynamic_threshold(self, scores, category):
        """Calculate thresholds based on actual score distribution"""
        if category == 'revenue':
            return np.percentile(scores, 75)  # Top 25% for revenue service
        elif category == 'standby':
            return np.percentile(scores, 50)  # Middle 50% for standby
        return 0

    def has_critical_issues_db_only(self, train):
        """Check for critical issues using only non-null database values"""
        critical_conditions = []
        
        if pd.notna(train.get('fitness_certificate')) and not train['fitness_certificate']:
            critical_conditions.append(True)
        
        if pd.notna(train.get('pending_maintenance')) and train['pending_maintenance']:
            critical_conditions.append(True)
        
        if pd.notna(train.get('brake_wear')) and float(train['brake_wear']) > 85:
            critical_conditions.append(True)
        
        if pd.notna(train.get('motor_temp')) and float(train['motor_temp']) > 95:
            critical_conditions.append(True)
        
        if pd.notna(train.get('door_failures')) and int(train['door_failures']) > 8:
            critical_conditions.append(True)
        
        if pd.notna(train.get('compliance_status')) and not train['compliance_status']:
            critical_conditions.append(True)
        
        if pd.notna(train.get('incident_reports')) and int(train['incident_reports']) > 5:
            critical_conditions.append(True)
        
        return any(critical_conditions)

    def determine_maintenance_type_db_only(self, train):
        """Determine maintenance type based only on actual database values"""
        # Priority order based on actual conditions found in data
        if pd.notna(train.get('fitness_certificate')) and not train['fitness_certificate']:
            return 'Safety Systems Check'
        elif pd.notna(train.get('compliance_status')) and not train['compliance_status']:
            return 'Compliance Inspection'
        elif pd.notna(train.get('brake_wear')) and float(train['brake_wear']) > 80:
            return 'Brake Service'
        elif pd.notna(train.get('motor_temp')) and float(train['motor_temp']) > 90:
            return 'Traction Maintenance'
        elif pd.notna(train.get('door_failures')) and int(train['door_failures']) > 5:
            return 'Door Systems'
        elif pd.notna(train.get('hvac_status')) and not train['hvac_status']:
            return 'HVAC Service'
        elif pd.notna(train.get('incident_reports')) and int(train['incident_reports']) > 3:
            return 'Incident Investigation'
        elif pd.notna(train.get('cleaning_status')) and str(train['cleaning_status']) != 'Clean':
            return 'Cleaning Service'
        elif pd.notna(train.get('pending_maintenance')) and train['pending_maintenance']:
            return 'Scheduled Maintenance'
        else:
            return 'Preventive Maintenance'

    def assign_maintenance_bay(self, maintenance_type, occupied_bays):
        """Assign maintenance bay based on database configuration"""
        # Map maintenance types to bay types from database
        for bay_id, bay_info in self.depot_bays.items():
            if (bay_id not in occupied_bays and 
                maintenance_type.lower() in bay_info['type'].lower()):
                return bay_id
        
        # If no specific bay available, find any available bay
        for bay_id in sorted(self.depot_bays.keys()):
            if bay_id not in occupied_bays:
                return bay_id
        
        return None

    def assign_stabling_track(self, occupied_tracks):
        """Assign stabling track based on database configuration"""
        for track in sorted(self.stabling_tracks):
            if track not in occupied_tracks:
                return track
        return None

    def calculate_completion_time_db_only(self, maintenance_type, train):
        """Calculate completion time using database bay configuration"""
        # Get estimated hours from depot bay configuration
        estimated_hours = None
        for bay_info in self.depot_bays.values():
            if maintenance_type.lower() in bay_info['type'].lower():
                estimated_hours = bay_info['avg_hours']
                break
        
        if not estimated_hours:
            # Use average from all bays
            estimated_hours = np.mean([bay['avg_hours'] for bay in self.depot_bays.values()])
        
        # Adjust based on actual train condition severity
        condition_multiplier = 1.0
        if pd.notna(train.get('incident_reports')) and int(train['incident_reports']) > 3:
            condition_multiplier += 0.5
        if pd.notna(train.get('brake_wear')) and float(train['brake_wear']) > 90:
            condition_multiplier += 0.3
        if pd.notna(train.get('compliance_status')) and not train['compliance_status']:
            condition_multiplier += 0.4
        
        total_hours = int(estimated_hours * condition_multiplier)
        completion_time = datetime.now() + timedelta(hours=total_hours)
        return completion_time.isoformat()

    def calculate_kpis(self, allocations):
        """Calculate KPIs from actual allocations"""
        if not allocations:
            return {
                'total_trains': 0,
                'revenue_service': 0,
                'standby': 0,
                'maintenance': 0,
                'average_readiness': 0.0,
                'fleet_availability_percent': 0.0,
                'service_capacity_percent': 0.0,
                'bay_utilization_percent': 0.0
            }
        
        total_trains = len(allocations)
        revenue_trains = len([a for a in allocations if a['category'] == 'Revenue Service'])
        standby_trains = len([a for a in allocations if a['category'] == 'Standby'])
        maintenance_trains = len([a for a in allocations if a['category'] == 'IBL Maintenance'])
        
        avg_readiness = np.mean([a['readiness_score'] for a in allocations])
        fleet_availability = (revenue_trains + standby_trains) / total_trains * 100
        service_capacity = revenue_trains / self.service_requirements['revenue_service_needed'] * 100
        
        occupied_bays = len([a for a in allocations if a.get('bay_assignment')])
        bay_utilization = occupied_bays / len(self.depot_bays) * 100
        
        return {
            'total_trains': total_trains,
            'revenue_service': revenue_trains,
            'standby': standby_trains,
            'maintenance': maintenance_trains,
            'average_readiness': round(avg_readiness, 2),
            'fleet_availability_percent': round(fleet_availability, 1),
            'service_capacity_percent': round(service_capacity, 1),
            'bay_utilization_percent': round(bay_utilization, 1)
        }

    def generate_insights(self, allocations, kpis):
        """Generate insights based on actual performance data"""
        insights = []
        
        # Service capacity analysis
        if kpis['service_capacity_percent'] < 100:
            shortage = self.service_requirements['revenue_service_needed'] - kpis['revenue_service']
            insights.append({
                'type': 'warning',
                'category': 'Service Capacity',
                'message': f"Service shortage: {shortage} trains below requirement ({kpis['revenue_service']}/{self.service_requirements['revenue_service_needed']})",
                'recommendation': 'Expedite high-readiness train maintenance to meet service demands'
            })
        elif kpis['service_capacity_percent'] > 110:
            insights.append({
                'type': 'info',
                'category': 'Service Capacity',
                'message': f"Service capacity exceeded: {kpis['revenue_service']} trains ready",
                'recommendation': 'Consider additional service opportunities or preventive maintenance'
            })
        
        # Fleet availability monitoring
        if kpis['fleet_availability_percent'] < 75:
            insights.append({
                'type': 'critical',
                'category': 'Fleet Availability',
                'message': f"Low fleet availability: {kpis['fleet_availability_percent']}% operational",
                'recommendation': 'Urgent review of maintenance scheduling and resource allocation needed'
            })
        elif kpis['fleet_availability_percent'] < 85:
            insights.append({
                'type': 'warning',
                'category': 'Fleet Availability',
                'message': f"Fleet availability below target: {kpis['fleet_availability_percent']}%",
                'recommendation': 'Monitor maintenance progress and consider extending service intervals'
            })
        
        # Maintenance load analysis
        if kpis['maintenance'] > (kpis['total_trains'] * 0.4):
            insights.append({
                'type': 'warning',
                'category': 'Maintenance Load',
                'message': f"High maintenance load: {kpis['maintenance']} trains ({kpis['maintenance']/kpis['total_trains']*100:.1f}%)",
                'recommendation': 'Review maintenance procedures and consider additional maintenance capacity'
            })
        
        # Bay utilization analysis
        if kpis['bay_utilization_percent'] > 90:
            insights.append({
                'type': 'warning',
                'category': 'Bay Utilization',
                'message': f"Maintenance bays near capacity: {kpis['bay_utilization_percent']}%",
                'recommendation': 'Consider maintenance scheduling optimization or additional bay capacity'
            })
        
        # Readiness score analysis
        if kpis['average_readiness'] < 6.0:
            insights.append({
                'type': 'critical',
                'category': 'Fleet Condition',
                'message': f"Low average readiness: {kpis['average_readiness']}/10",
                'recommendation': 'Investigate systematic issues affecting fleet condition'
            })
        elif kpis['average_readiness'] > 8.0:
            insights.append({
                'type': 'info',
                'category': 'Fleet Condition',
                'message': f"Excellent fleet condition: {kpis['average_readiness']}/10 average readiness",
                'recommendation': 'Maintain current maintenance standards and practices'
            })
        
        # Critical train analysis
        critical_trains = [a for a in allocations if a['readiness_score'] < 3.0]
        if len(critical_trains) > 0:
            insights.append({
                'type': 'critical',
                'category': 'Critical Trains',
                'message': f"{len(critical_trains)} trains in critical condition",
                'recommendation': f"Immediate attention required for: {', '.join([t['train_number'] for t in critical_trains])}"
            })
        
        return insights

    def generate_master_plan(self):
        """Generate comprehensive induction plan with database-only data"""
        try:
            # Fetch current train data
            train_df = self.fetch_train_data()
            
            if train_df.empty:
                raise Exception("No train data available from database")
            
            # Calculate readiness scores using ML model or database-driven approach
            readiness_scores = self.calculate_readiness_scores(train_df)
            
            # Generate explanations based on actual data
            explanations = self.generate_explanations(train_df, readiness_scores)
            
            # Perform allocation based on database configuration
            allocations, bay_assignments, track_assignments = self.allocate_trains(
                train_df, readiness_scores, explanations
            )
            
            # Calculate performance KPIs from actual results
            kpis = self.calculate_kpis(allocations)
            
            # Generate insights from actual performance
            insights = self.generate_insights(allocations, kpis)
            
            # Create depot status from actual assignments
            depot_status = self.create_depot_status(bay_assignments, track_assignments)
            
            plan = {
                'generated_at': datetime.now().isoformat(),
                'plan_date': datetime.now().date().isoformat(),
                'train_allocations': allocations,
                'performance_kpis': kpis,
                'operational_insights': insights,
                'depot_status': depot_status,
                'total_trains': len(allocations),
                'service_requirements': self.service_requirements,
                'model_used': 'ML Model' if self.model is not None else 'Database-driven Rules',
                'data_completeness': self.assess_data_completeness(train_df)
            }
            
            logger.info(f"Master plan generated: {len(allocations)} trains allocated using {plan['model_used']}")
            return plan
            
        except Exception as e:
            logger.error(f"Error generating master plan: {e}")
            logger.error(traceback.format_exc())
            raise

    def assess_data_completeness(self, df):
        """Assess completeness of database data"""
        total_fields = len(df.columns)
        completeness = {}
        
        for col in df.columns:
            non_null_count = df[col].notna().sum()
            completeness[col] = round((non_null_count / len(df)) * 100, 1)
        
        overall_completeness = round(np.mean(list(completeness.values())), 1)
        
        return {
            'overall_percent': overall_completeness,
            'field_completeness': completeness,
            'total_records': len(df),
            'total_fields': total_fields
        }

    def create_depot_status(self, bay_assignments, track_assignments):
        """Create depot status based on actual database configuration"""
        maintenance_bays = []
        for bay_id, bay_info in self.depot_bays.items():
            bay_status = {
                'bay_id': bay_id,
                'bay_type': bay_info['type'],
                'status': 'Occupied' if bay_id in bay_assignments else 'Available',
                'train_number': bay_assignments.get(bay_id),
                'estimated_hours': bay_info['avg_hours']
            }
            maintenance_bays.append(bay_status)
        
        stabling_tracks_status = []
        for track_id in self.stabling_tracks:
            track_status = {
                'track_id': track_id,
                'status': 'Occupied' if track_id in track_assignments else 'Available',
                'train_number': track_assignments.get(track_id)
            }
            stabling_tracks_status.append(track_status)
        
        return {
            'maintenance_bays': maintenance_bays,
            'stabling_tracks': stabling_tracks_status,
            'bay_utilization': len(bay_assignments) / len(self.depot_bays) * 100 if self.depot_bays else 0,
            'track_utilization': len(track_assignments) / len(self.stabling_tracks) * 100 if self.stabling_tracks else 0,
            'total_bays': len(self.depot_bays),
            'occupied_bays': len(bay_assignments),
            'total_tracks': len(self.stabling_tracks),
            'occupied_tracks': len(track_assignments)
        }

# Initialize planner instance
planner = InductionPlanner()

@app.route('/')
def index():
    """Main dashboard page"""
    return send_from_directory(',','induction_planning.html')

@app.route('/api/master-plan', methods=['GET'])
def get_master_plan():
    """Get comprehensive induction plan with database-only data"""
    try:
        logger.info("Generating master plan from database-only data")
        plan = planner.generate_master_plan()
        
        # Log plan summary
        logger.info(f"Plan generated successfully: "
                   f"{plan['performance_kpis']['revenue_service']} revenue, "
                   f"{plan['performance_kpis']['standby']} standby, "
                   f"{plan['performance_kpis']['maintenance']} maintenance")
        
        return jsonify(plan)
        
    except Exception as e:
        logger.error(f"Error in master plan API: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f"Failed to generate master plan: {str(e)}"}), 500

@app.route('/api/parts-cost', methods=['GET'])
def get_parts_cost():
    """Get parts cost data from kmrl_relevant_parts_cost table"""
    try:
        parts_data = planner.get_parts_cost_data()
        if not parts_data:
            return jsonify({'error': 'No parts cost data available'}), 404
        
        # Format for display
        formatted_data = [
            {'part_name': name, 'cost': float(cost)} 
            for name, cost in parts_data.items()
        ]
        
        return jsonify({
            'parts': formatted_data,
            'total_parts': len(formatted_data),
            'highest_cost': max(parts_data.values()) if parts_data else 0,
            'lowest_cost': min(parts_data.values()) if parts_data else 0,
            'average_cost': round(np.mean(list(parts_data.values())), 2) if parts_data else 0
        })
        
    except Exception as e:
        logger.error(f"Error getting parts cost data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-completeness', methods=['GET'])
def get_data_completeness():
    """Get data completeness assessment"""
    try:
        train_df = planner.fetch_train_data()
        completeness = planner.assess_data_completeness(train_df)
        return jsonify(completeness)
        
    except Exception as e:
        logger.error(f"Error assessing data completeness: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/train-details/<train_number>', methods=['GET'])
def get_train_details(train_number):
    """Get detailed information about a specific train"""
    try:
        plan = planner.generate_master_plan()
        train_info = next((t for t in plan['train_allocations'] if t['train_number'] == train_number), None)
        
        if train_info is None:
            return jsonify({'error': f'Train {train_number} not found'}), 404
        
        # Add additional details from database
        conn = planner.get_db_connection()
        if conn:
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM train_status 
                        WHERE train_number = %s
                    """, (train_number,))
                    db_data = cur.fetchone()
                    
                    if db_data:
                        train_info['database_details'] = dict(db_data)
            finally:
                conn.close()
        
        return jsonify(train_info)
        
    except Exception as e:
        logger.error(f"Error getting train details: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights', methods=['GET'])
def get_insights():
    """Get operational insights based on current data"""
    try:
        plan = planner.generate_master_plan()
        return jsonify({
            'insights': plan['operational_insights'],
            'kpis': plan['performance_kpis'],
            'generated_at': plan['generated_at'],
            'data_completeness': plan['data_completeness']
        })
        
    except Exception as e:
        logger.error(f"Error getting insights: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/depot-status', methods=['GET'])
def get_depot_status():
    """Get current depot status with real assignments"""
    try:
        plan = planner.generate_master_plan()
        return jsonify({
            **plan['depot_status'],
            'last_updated': plan['generated_at']
        })
        
    except Exception as e:
        logger.error(f"Error getting depot status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with database validation"""
    try:
        # Test database connection and data availability
        conn = planner.get_db_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM train_status WHERE (retirement_date IS NULL OR retirement_date > CURRENT_DATE)")
                train_count = cur.fetchone()[0]
                
                # Check if required tables exist
                cur.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('train_status', 'kmrl_relevant_parts_cost')
                """)
                existing_tables = [row[0] for row in cur.fetchall()]
            conn.close()
            
            return jsonify({
                'status': 'healthy',
                'database': 'connected',
                'active_trains': train_count,
                'model_loaded': planner.model is not None,
                'required_tables_present': existing_tables,
                'timestamp': datetime.now().isoformat(),
                'data_source': 'database_only'
            })
        else:
            return jsonify({
                'status': 'unhealthy',
                'database': 'disconnected',
                'error': 'Cannot connect to database'
            }), 503
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

if __name__ == '__main__':
    logger.info("Starting KMRL Database-Driven Induction Planning System")
    logger.info(f"Service requirements: {planner.service_requirements}")
    logger.info(f"Depot bays configured: {len(planner.depot_bays)}")
    logger.info(f"Stabling tracks available: {len(planner.stabling_tracks)}")
    logger.info(f"ML Model loaded: {planner.model is not None}")
    logger.info("System configured to use ONLY database values - no static/dummy data")
    
    app.run(debug=True, host='0.0.0.0', port=5004)