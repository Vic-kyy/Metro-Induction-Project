from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Direct Supabase PostgreSQL connection
DB_CONN = "postgresql://postgres:RkoGkPLWxh4vavX3@db.trwsfdhxzwzkjandsmvz.supabase.co:5432/postgres"

# Global variables
model = None
scaler = None
feature_columns = None
model_feature_names = None
db_connection_status = False

def load_models():
    """Load the trained model and scaler"""
    global model, scaler, feature_columns, model_feature_names
    
    try:
        model_paths = [
            ('realistic_kochi_metro_rf_model.pkl', 'realistic_kochi_metro_scaler.pkl'),
            ('models/realistic_kochi_metro_rf_model.pkl', 'models/realistic_kochi_metro_scaler.pkl'),
        ]
        
        for model_path, scaler_path in model_paths:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                # Get actual feature names from the scaler or model
                if hasattr(scaler, 'feature_names_in_'):
                    model_feature_names = list(scaler.feature_names_in_)
                    logger.info(f"Found {len(model_feature_names)} feature names from scaler")
                else:
                    # Fallback to defined feature columns
                    model_feature_names = [
                        'mtbf', 'brake_wear', 'energy_kwh', 'fault_code', 'mileage_km', 
                        'motor_temp', 'trip_count', 'energy_cost', 'hvac_status', 
                        'door_failures', 'available_bays', 'depot_location', 'depot_position',
                        'battery_current', 'battery_voltage', 'cleaning_status', 'operating_hours',
                        'door_cycle_count', 'incident_reports', 'maintenance_cost', 
                        'compliance_status', 'work_order_status', 'passenger_capacity',
                        'passengers_onboard', 'depot_accessibility', 'fitness_certificate',
                        'pending_maintenance', 'standby_requirement', 'total_operating_cost',
                        'operating_cost_per_km', 'advertising_commitments', 
                        'operating_cost_per_hour', 'stabling_geometry_score',
                        'occupancy_ratio', 'energy_per_km', 'mileage_balancing'
                    ]
                
                feature_columns = model_feature_names.copy()
                
                logger.info(f"Models loaded successfully from {model_path}")
                logger.info(f"Model type: {type(model).__name__}")
                logger.info(f"Expected features: {len(feature_columns)}")
                if hasattr(model, 'n_estimators'):
                    logger.info(f"Random Forest with {model.n_estimators} estimators")
                
                return True
        
        logger.warning("Model files not found - ML predictions not available")
        return False
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

def test_db_connection():
    """Test database connection"""
    global db_connection_status
    try:
        conn = psycopg2.connect(DB_CONN)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM train_status")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        db_connection_status = True
        logger.info(f"Database connected successfully - {count} trains found")
        return True
    except Exception as e:
        db_connection_status = False
        logger.error(f"Database connection failed: {str(e)}")
        return False

def get_trains_from_db():
    """Fetch all train data from database - NO FALLBACK DATA"""
    if not db_connection_status:
        logger.error("Database not connected - cannot fetch trains")
        return []
        
    try:
        conn = psycopg2.connect(DB_CONN)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT train_number, mtbf, brake_wear, energy_kwh, fault_code, mileage_km, 
                   motor_temp, trip_count, energy_cost, hvac_status, door_failures, 
                   available_bays, depot_location, depot_position, battery_current, 
                   battery_voltage, cleaning_status, operating_hours, retirement_date, 
                   door_cycle_count, incident_reports, maintenance_cost, compliance_status,
                   mileage_balancing, work_order_status, passenger_capacity, passengers_onboard,
                   depot_accessibility, fitness_certificate, pending_maintenance,
                   standby_requirement, total_operating_cost, last_maintenance_date,
                   operating_cost_per_km, advertising_commitments, operating_cost_per_hour,
                   stabling_geometry_score, occupancy_ratio, energy_per_km
            FROM train_status 
            ORDER BY train_number
        """)
        
        trains = cur.fetchall()
        cur.close()
        conn.close()
        
        if trains:
            logger.info(f"Retrieved {len(trains)} trains from Supabase database")
            return [dict(train) for train in trains]
        else:
            logger.warning("No trains found in database")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching trains from database: {str(e)}")
        return []

def get_train_by_number(train_number):
    """Fetch specific train data from database - NO FALLBACK DATA"""
    if not db_connection_status:
        logger.error("Database not connected - cannot fetch train")
        return None
        
    try:
        conn = psycopg2.connect(DB_CONN)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT train_number, mtbf, brake_wear, energy_kwh, fault_code, mileage_km, 
                   motor_temp, trip_count, energy_cost, hvac_status, door_failures, 
                   available_bays, depot_location, depot_position, battery_current, 
                   battery_voltage, cleaning_status, operating_hours, retirement_date, 
                   door_cycle_count, incident_reports, maintenance_cost, compliance_status,
                   mileage_balancing, work_order_status, passenger_capacity, passengers_onboard,
                   depot_accessibility, fitness_certificate, pending_maintenance,
                   standby_requirement, total_operating_cost, last_maintenance_date,
                   operating_cost_per_km, advertising_commitments, operating_cost_per_hour,
                   stabling_geometry_score, occupancy_ratio, energy_per_km
            FROM train_status 
            WHERE train_number = %s
        """, (train_number,))
        
        train = cur.fetchone()
        cur.close()
        conn.close()
        
        if train:
            logger.info(f"Retrieved train {train_number} from database")
            return dict(train)
        else:
            logger.warning(f"Train {train_number} not found in database")
            return None
            
    except Exception as e:
        logger.error(f"Error fetching train {train_number}: {str(e)}")
        return None

def validate_input_data(data):
    """Validate essential input data"""
    required_fields = ['train_number']
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == '':
            raise ValueError(f"Missing required field: {field}")
    return True

def preprocess_data(data):
    """Preprocess input data for ML model"""
    try:
        processed_data = data.copy()
        
        # Label encoders (same as training)
        label_encoders = {
            'fault_code': {'': 0, 'F01': 1, 'F02': 2, 'F03': 3, 'F04': 4, 'F05': 5},
            'depot_location': {'Aluva': 0, 'Edappally': 1, 'Mannuthy': 2},
            'depot_position': {'A1': 0, 'B2': 1, 'C3': 2, 'D4': 3},
            'cleaning_status': {'Clean': 0, 'In-progress': 1, 'Pending': 2},
            'work_order_status': {'Closed': 0, 'Open': 1},
            'depot_accessibility': {'Difficult': 0, 'Easy': 1, 'Moderate': 2},
            'advertising_commitments': {'High': 0, 'Low': 1, 'Medium': 2}
        }
        
        # Apply label encoding
        for column, encoding_map in label_encoders.items():
            if column in processed_data and processed_data[column] is not None:
                processed_data[column] = encoding_map.get(str(processed_data[column]), 0)
        
        # Convert boolean columns
        bool_columns = ['hvac_status', 'compliance_status', 'fitness_certificate',
                       'pending_maintenance', 'standby_requirement']
        for col in bool_columns:
            if col in processed_data:
                if isinstance(processed_data[col], bool):
                    processed_data[col] = int(processed_data[col])
                elif str(processed_data[col]).lower() in ['true', '1', 'yes']:
                    processed_data[col] = 1
                else:
                    processed_data[col] = 0
        
        # Fill missing values with sensible defaults only for ML prediction
        defaults = {
            'trip_count': 25,
            'energy_cost': 500,
            'hvac_status': 1,
            'battery_current': 250,
            'battery_voltage': 700,
            'compliance_status': 1,
            'standby_requirement': 0,
            'door_failures': 0,
            'operating_hours': 8760,
            'door_cycle_count': 50000,
            'incident_reports': 0,
            'maintenance_cost': 50000,
            'total_operating_cost': 200000,
            'operating_cost_per_km': 8.0,
            'operating_cost_per_hour': 25.0,
            'passenger_capacity': 600,
            'passengers_onboard': 300
        }
        
        for key, default_val in defaults.items():
            if key not in processed_data or processed_data[key] is None:
                processed_data[key] = default_val
        
        # Calculate derived features only if missing
        if 'occupancy_ratio' not in processed_data or processed_data['occupancy_ratio'] is None:
            passengers = processed_data.get('passengers_onboard', 300)
            capacity = processed_data.get('passenger_capacity', 600)
            processed_data['occupancy_ratio'] = passengers / max(capacity, 1)
        
        if 'energy_per_km' not in processed_data or processed_data['energy_per_km'] is None:
            energy = processed_data.get('energy_kwh', 500)
            mileage = processed_data.get('mileage_km', 25000)
            processed_data['energy_per_km'] = energy / max(mileage, 1)
        
        if 'mileage_balancing' not in processed_data or processed_data['mileage_balancing'] is None:
            brake_wear = processed_data.get('brake_wear', 30)
            mileage_km = processed_data.get('mileage_km', 25000)
            processed_data['mileage_balancing'] = 1 - (brake_wear/100 * 0.5 + mileage_km/50000 * 0.5)
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def predict_with_model(data):
    """Make prediction using the trained model - ONLY when model is available"""
    if not model or not scaler:
        raise ValueError("ML model not loaded - cannot make ML prediction")
    
    try:
        # Validate input
        validate_input_data(data)
        
        # Preprocess data
        processed_data = preprocess_data(data)
        
        # Create DataFrame with all expected features
        df_data = {}
        
        # Fill all required features
        for feature_name in feature_columns:
            if feature_name in processed_data:
                df_data[feature_name] = processed_data[feature_name]
            else:
                # Provide sensible defaults for missing features
                defaults = {
                    'stabling_geometry_score': 0.7,
                    'advertising_commitments': 1,
                    'depot_location': 0,
                    'depot_position': 0,
                    'cleaning_status': 0,
                    'compliance_status': 1,
                    'depot_accessibility': 1,
                    'work_order_status': 0,
                    'fault_code': 0,
                    'hvac_status': 1,
                    'fitness_certificate': 1,
                    'pending_maintenance': 0,
                    'standby_requirement': 0
                }
                df_data[feature_name] = defaults.get(feature_name, 0)
        
        # Create DataFrame and select features in exact order
        df = pd.DataFrame([df_data])
        X = df[feature_columns]
        
        logger.debug(f"Input shape: {X.shape}")
        
        # Convert to numpy array to avoid sklearn feature name validation issues
        X_array = X.values
        X_scaled = scaler.transform(X_array)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        # Get feature importance
        feature_contributions = []
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            
            for i, (feature, importance) in enumerate(zip(feature_columns, feature_importances)):
                contribution = {
                    'feature': feature,
                    'importance': float(importance),
                    'value': float(X.iloc[0, i])
                }
                feature_contributions.append(contribution)
            
            feature_contributions.sort(key=lambda x: x['importance'], reverse=True)
        
        # Determine category
        if prediction >= 7.5:
            category, category_class = 'Revenue Service Ready', 'revenue-service'
        elif prediction >= 6.0:
            category, category_class = 'Standby Status', 'standby'
        else:
            category, category_class = 'IBL Maintenance Required', 'maintenance'
        
        logger.info(f"ML prediction successful: {prediction:.2f} -> {category}")
        
        return {
            'prediction': float(prediction),
            'category': category,
            'category_class': category_class,
            'feature_contributions': feature_contributions[:10],
            'prediction_method': 'ML_MODEL'
        }
        
    except Exception as e:
        logger.error(f"Error in predict_with_model: {str(e)}")
        raise

@app.route('/')
def home():
    """Serve the dashboard"""
    try:
        if os.path.exists('kochi_metro_whatif.html'):
            return send_from_directory('.', 'kochi_metro_whatif.html')
        else:
            return jsonify({
                'message': 'Kochi Metro AI Backend is Running',
                'status': 'active',
                'database_connected': db_connection_status,
                'model_loaded': model is not None,
                'endpoints': {
                    'GET /whatif/trains': 'Get all trains from Supabase database',
                    'GET /whatif/train/<train_number>': 'Get specific train from database',
                    'POST /whatif/predict': 'Predict with manual overrides only',
                    'GET /health': 'Health check',
                    'GET /fleet': 'Fleet overview from database'
                }
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with accurate status"""
    try:
        # Test database connection
        db_status = test_db_connection()
        
        if db_status:
            conn = psycopg2.connect(DB_CONN)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM train_status")
            train_count = cur.fetchone()[0]
            cur.close()
            conn.close()
        else:
            train_count = 0
        
        return jsonify({
            'status': 'healthy' if db_status else 'degraded',
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'database_connected': db_status,
            'trains_in_db': train_count,
            'expected_features': len(feature_columns) if feature_columns else 0,
            'data_source': 'supabase_only',
            'simulation_mode': 'manual_overrides_only',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'database_connected': False,
            'trains_in_db': 0,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/whatif/trains', methods=['GET'])
def get_all_trains():
    """Get all trains from Supabase database only - NO FALLBACK"""
    try:
        if not db_connection_status:
            return jsonify({
                'error': 'Database not connected',
                'message': 'Cannot load train data - Supabase database is not available'
            }), 503
            
        trains = get_trains_from_db()
        
        if not trains:
            return jsonify({
                'error': 'No trains found',
                'message': 'Database is connected but contains no train records'
            }), 404
            
        return jsonify(trains)
        
    except Exception as e:
        logger.error(f"Error in get_all_trains: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Failed to retrieve trains from database'
        }), 500

@app.route('/whatif/train/<train_number>', methods=['GET'])
def get_train(train_number):
    """Get specific train data from Supabase database only - NO FALLBACK"""
    try:
        if not db_connection_status:
            return jsonify({
                'error': 'Database not connected',
                'message': 'Cannot load train data - Supabase database is not available'
            }), 503
            
        train = get_train_by_number(train_number)
        
        if train:
            return jsonify(train)
        else:
            return jsonify({
                'error': 'Train not found',
                'message': f'Train {train_number} not found in database'
            }), 404
            
    except Exception as e:
        logger.error(f"Error in get_train: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': f'Failed to retrieve train {train_number}'
        }), 500

@app.route('/whatif/predict', methods=['POST'])
def predict_whatif():
    """Main prediction endpoint - Database + Manual Overrides ONLY"""
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'error': 'No data provided'}), 400
        
        train_number = request_data.get('train_number')
        if not train_number:
            return jsonify({'error': 'train_number is required'}), 400
        
        if not db_connection_status:
            return jsonify({
                'error': 'Database not connected',
                'message': 'Cannot perform analysis - Supabase database is not available'
            }), 503
        
        # Get base data from Supabase database ONLY
        base_train_data = get_train_by_number(train_number)
        
        if not base_train_data:
            return jsonify({
                'error': 'Train not found in database',
                'message': f'Train {train_number} does not exist in Supabase database'
            }), 404
        
        logger.info(f"Base data loaded from Supabase for {train_number}")
        final_data = base_train_data.copy()
        data_source = 'database'
        
        # Apply ONLY the manual overrides from the request
        override_fields = [
            'mtbf', 'brake_wear', 'energy_kwh', 'fault_code', 'mileage_km',
            'motor_temp', 'available_bays', 'fitness_certificate', 'pending_maintenance',
            'cleaning_status', 'stabling_geometry_score', 'trip_count', 'energy_cost',
            'hvac_status', 'door_failures', 'depot_location', 'depot_position',
            'battery_current', 'battery_voltage', 'operating_hours', 'door_cycle_count',
            'incident_reports', 'maintenance_cost', 'compliance_status', 'work_order_status',
            'passenger_capacity', 'passengers_onboard', 'depot_accessibility',
            'standby_requirement', 'total_operating_cost', 'operating_cost_per_km',
            'advertising_commitments', 'operating_cost_per_hour'
        ]
        
        overrides_applied = []
        for field in override_fields:
            if field in request_data:
                original_value = final_data.get(field)
                new_value = request_data[field]
                
                # Only apply if values are different
                if original_value != new_value:
                    final_data[field] = new_value
                    overrides_applied.append({
                        'field': field,
                        'original': original_value,
                        'override': new_value
                    })
                    logger.debug(f"Override applied: {field} = {new_value} (was {original_value})")
        
        # Ensure train_number is preserved
        final_data['train_number'] = train_number
        
        # Make prediction based on available methods
        try:
            if model and scaler:
                # Use ML model if available
                result = predict_with_model(final_data)
                prediction_method = 'ML_MODEL'
                logger.info(f"ML prediction completed for {train_number}: {result['prediction']:.2f}")
            else:
                # No simulation fallback - return error if model not available
                return jsonify({
                    'error': 'ML model not available',
                    'message': 'Cannot perform prediction - machine learning model is not loaded'
                }), 503
                
        except Exception as pred_error:
            logger.error(f"ML prediction failed: {str(pred_error)}")
            return jsonify({
                'error': 'Prediction failed',
                'message': str(pred_error)
            }), 500
        
        # Add metadata
        result['train_number'] = train_number
        result['data_source'] = data_source
        result['overrides_applied'] = len(overrides_applied)
        result['override_details'] = overrides_applied
        result['prediction_method'] = prediction_method
        result['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Prediction successful for {train_number} with {len(overrides_applied)} overrides")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Prediction failed due to system error'
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': type(model).__name__ if model else 'Not loaded',
        'n_estimators': getattr(model, 'n_estimators', 'N/A'),
        'max_depth': getattr(model, 'max_depth', 'N/A'),
        'feature_count': len(feature_columns) if feature_columns else 0,
        'feature_names': feature_columns[:10] if feature_columns else [],
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'database_connected': db_connection_status,
        'data_source': 'supabase_only',
        'simulation_mode': 'disabled'
    })

@app.route('/fleet', methods=['GET'])
def get_fleet():
    """Get fleet overview with health scores from database only - USE APP.PY LOGIC"""
    try:
        if not db_connection_status:
            return jsonify({
                'error': 'Database not connected',
                'message': 'Cannot load fleet data - Supabase database is not available'
            }), 503
            
        # Import the app.py logic for ML predictions
        from app import get_fleet as app_get_fleet
        
        # Use the existing app.py fleet logic which has proper ML integration
        try:
            # Call the app.py get_fleet function directly
            app_response = app_get_fleet()
            
            # Extract the JSON data from the Flask response
            if hasattr(app_response, 'get_json'):
                fleet_data = app_response.get_json()
            else:
                # Handle direct JSON response
                import json
                fleet_data = json.loads(app_response.data.decode('utf-8'))
            
            logger.info(f"Fleet overview generated using app.py ML logic for {len(fleet_data)} trains")
            return jsonify(fleet_data)
            
        except ImportError:
            logger.warning("Could not import app.py - using fallback logic")
            return get_fleet_fallback()
            
    except Exception as e:
        logger.error(f"Error in get_fleet: {str(e)}")
        return get_fleet_fallback()

def get_fleet_fallback():
    """Fallback fleet calculation when app.py is not available"""
    try:
        trains = get_trains_from_db()
        
        if not trains:
            return jsonify({
                'error': 'No fleet data found',
                'message': 'Database is connected but contains no train records'
            }), 404
        
        # Calculate health scores using basic rules if ML model not available
        fleet_results = []
        for train_data in trains:
            try:
                if model and scaler:
                    # Use ML model to calculate health from actual data
                    result = predict_with_model(train_data)
                    health = int(result['prediction'] * 10)
                else:
                    # Basic rule-based health calculation from database data
                    health = calculate_basic_health_score(train_data)
                
                # Determine status based on health (using app.py logic)
                if health >= 80:
                    status = 'Active'
                elif health >= 60:
                    status = 'Alert'
                else:
                    status = 'Maintenance'
                
                # Calculate next maintenance based on actual data
                last_maintenance = train_data.get('last_maintenance_date')
                if last_maintenance:
                    try:
                        last_date = datetime.strptime(str(last_maintenance), '%Y-%m-%d')
                        days_since = (datetime.now() - last_date).days
                        next_maintenance_days = max(1, 30 - days_since + (health - 50) // 5)
                    except:
                        next_maintenance_days = max(1, health // 3)
                else:
                    next_maintenance_days = max(1, health // 3)
                
                next_maintenance_date = (datetime.now() + timedelta(days=next_maintenance_days)).strftime('%Y-%m-%d')
                
                fleet_results.append({
                    'id': train_data['train_number'],
                    'status': status,
                    'health': health,
                    'mileage': int(train_data.get('mileage_km', 0)),
                    'nextMaintenance': next_maintenance_date,
                    'brake_wear': train_data.get('brake_wear', 0),
                    'mtbf': train_data.get('mtbf', 0)
                })
                
            except Exception as e:
                logger.error(f"Error processing train {train_data.get('train_number', 'unknown')}: {e}")
                continue
        
        if not fleet_results:
            return jsonify({
                'error': 'Fleet processing failed',
                'message': 'Could not process any train data for fleet overview'
            }), 500
        
        logger.info(f"Fleet overview generated for {len(fleet_results)} trains from Supabase (fallback)")
        return jsonify(fleet_results)
        
    except Exception as e:
        logger.error(f"Error in get_fleet_fallback: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Failed to generate fleet overview'
        }), 500

def calculate_basic_health_score(train_data):
    """Calculate basic health score from database data without ML model"""
    score = 50  # Base score
    
    # Fitness certificate
    if train_data.get('fitness_certificate', False):
        score += 20
    else:
        score -= 25
    
    # Pending maintenance
    if not train_data.get('pending_maintenance', False):
        score += 15
    else:
        score -= 20
    
    # Brake wear
    brake_wear = train_data.get('brake_wear', 50)
    if brake_wear < 30:
        score += 10
    elif brake_wear > 70:
        score -= 15
    
    # MTBF
    mtbf = train_data.get('mtbf', 1500)
    if mtbf > 2000:
        score += 10
    elif mtbf < 1000:
        score -= 10
    
    # Mileage
    mileage = train_data.get('mileage_km', 25000)
    if mileage < 20000:
        score += 5
    elif mileage > 40000:
        score -= 10
    
    # Work orders
    if train_data.get('work_order_status') == 'Closed':
        score += 5
    
    # Ensure score is within bounds
    return max(10, min(100, score))

if __name__ == '__main__':
    try:
        logger.info("Starting Kochi Metro What-If Analysis Backend...")
        
        # Test database connection first
        db_connected = test_db_connection()
        if db_connected:
            logger.info("✅ Supabase database connected successfully")
        else:
            logger.error("❌ Supabase database connection failed - limited functionality")
        
        # Load ML models
        model_loaded = load_models()
        if model_loaded:
            logger.info("✅ ML models loaded successfully")
        else:
            logger.warning("⚠️  ML models not loaded - predictions not available")
        
        if not db_connected and not model_loaded:
            logger.error("❌ Neither database nor ML models are available - backend cannot function")
        elif not db_connected:
            logger.error("❌ Database not connected - cannot load train data from Supabase")
        elif not model_loaded:
            logger.warning("⚠️  ML model not available - using basic health calculations")
        else:
            logger.info("✅ Backend fully operational with database and ML model")
        
        logger.info("Backend configuration:")
        logger.info("  - Data Source: Supabase PostgreSQL ONLY")
        logger.info("  - Simulation Mode: DISABLED")
        logger.info("  - Manual Overrides: ENABLED")
        logger.info("  - Fallback Data: DISABLED")
        
        logger.info("Frontend available at: http://localhost:5005")
        logger.info("API endpoints: /health, /whatif/trains, /whatif/predict, /fleet")
        
        app.run(
            host='0.0.0.0',
            port=5005,
            debug=True,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        print(f"Error: {str(e)}")
        print("Setup checklist:")
        print("1. Ensure Supabase database is accessible")
        print("2. Verify train_status table exists with data")
        print("3. Check that .pkl model files are available (optional)")
        print("4. Run database.py to populate data if needed")