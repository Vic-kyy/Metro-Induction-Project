from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv
from datetime import datetime, date, timedelta
import socket
import ssl
import logging
import traceback

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration - Same pattern as induction planner
DB_CONFIG = {
    'host': 'db.trwsfdhxzwzkjandsmvz.supabase.co',
    'database': 'postgres',
    'user': 'postgres',
    'password': 'RkoGkPLWxh4vavX3',
    'port': '5432'
}

class CleaningManager:
    def __init__(self):
        self.ensure_database_setup()
    
    def get_db_connection(self):
        """Get database connection with proper error handling - Same as induction planner"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            return conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            return None
    
    def test_connectivity(self):
        """Test network connectivity to database"""
        try:
            hostname = DB_CONFIG['host']
            port = int(DB_CONFIG['port'])
            
            # Test DNS resolution
            ip = socket.gethostbyname(hostname)
            logger.info(f"DNS resolved {hostname} to {ip}")
            
            # Test port connectivity
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((hostname, port))
            sock.close()
            
            if result == 0:
                logger.info(f"Port {port} is open on {hostname}")
                return True
            else:
                logger.warning(f"Cannot connect to port {port} on {hostname}")
                return False
                
        except socket.gaierror as e:
            logger.error(f"DNS resolution failed for {hostname}: {e}")
            return False
        except Exception as e:
            logger.error(f"Connectivity test failed: {e}")
            return False
    
    def ensure_database_setup(self):
        """Initialize database tables with comprehensive sample data"""
        conn = self.get_db_connection()
        if not conn:
            logger.error("Cannot initialize database - no connection")
            return False
        
        try:
            with conn.cursor() as cursor:
                # Create cleaning table if it doesn't exist
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cleaning (
                        id SERIAL PRIMARY KEY,
                        train_id VARCHAR(20) UNIQUE NOT NULL,
                        status_record VARCHAR(50) DEFAULT 'Pending',
                        computed_status VARCHAR(50),
                        last_sanitized TIMESTAMP,
                        next_schedule TIMESTAMP,
                        team_id VARCHAR(20),
                        time_slot_start TIME,
                        time_slot_end TIME,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_cleaning_train_id ON cleaning(train_id);
                    CREATE INDEX IF NOT EXISTS idx_cleaning_status ON cleaning(status_record);
                    CREATE INDEX IF NOT EXISTS idx_cleaning_team_id ON cleaning(team_id);
                ''')
                
                # Check if table has data
                cursor.execute("SELECT COUNT(*) FROM cleaning")
                count = cursor.fetchone()[0]
                logger.info(f"Current records in cleaning table: {count}")
                
                # Insert comprehensive sample data if table is empty or has few records
                if count < 20:
                    if count > 0:
                        logger.info(f"Found only {count} records, clearing and reinserting full dataset...")
                        cursor.execute("DELETE FROM cleaning")
                    else:
                        logger.info("Inserting comprehensive sample data...")
                    
                    # Generate realistic sample data for 25 trains with current dates
                    base_date = datetime.now()
                    
                    sample_data = [
                        ('KMR-1001', 'Scheduled', None, (base_date - timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S'), (base_date + timedelta(hours=10)).strftime('%Y-%m-%d %H:%M:%S'), 'T1', '12:15:00', '16:15:00'),
                        ('KMR-1002', 'Sanitized', None, (base_date - timedelta(hours=14)).strftime('%Y-%m-%d %H:%M:%S'), (base_date - timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S'), 'T3', '16:15:00', '17:15:00'),
                        ('KMR-1003', 'Pending', None, (base_date - timedelta(hours=18)).strftime('%Y-%m-%d %H:%M:%S'), (base_date - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'), 'T4', '12:30:00', '14:30:00'),
                        ('KMR-1004', 'In Progress', None, (base_date - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M:%S'), (base_date + timedelta(hours=11)).strftime('%Y-%m-%d %H:%M:%S'), 'T1', '07:30:00', '09:30:00'),
                        ('KMR-1005', 'Scheduled', None, (base_date - timedelta(hours=12)).strftime('%Y-%m-%d %H:%M:%S'), (base_date - timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S'), 'T5', '07:45:00', '10:45:00'),
                        ('KMR-1006', 'Scheduled', None, (base_date - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S'), (base_date - timedelta(hours=0, minutes=15)).strftime('%Y-%m-%d %H:%M:%S'), 'T5', '11:45:00', '13:45:00'),
                        ('KMR-1007', 'Sanitized', None, (base_date - timedelta(hours=17)).strftime('%Y-%m-%d %H:%M:%S'), (base_date - timedelta(hours=14)).strftime('%Y-%m-%d %H:%M:%S'), 'T3', '09:00:00', '13:00:00'),
                        ('KMR-1008', 'Sanitized', None, (base_date - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'), (base_date + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'), 'T4', '15:00:00', '17:00:00'),
                        ('KMR-1009', 'Scheduled', None, (base_date - timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S'), (base_date + timedelta(hours=13)).strftime('%Y-%m-%d %H:%M:%S'), 'T4', '13:00:00', '15:00:00'),
                        ('KMR-1010', 'Sanitized', None, (base_date - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'), (base_date + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'), 'T3', '10:00:00', '11:00:00'),
                        ('KMR-1011', 'Sanitized', None, (base_date - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S'), (base_date + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'), 'T4', '15:30:00', '16:30:00'),
                        ('KMR-1012', 'Sanitized', None, (base_date - timedelta(hours=15)).strftime('%Y-%m-%d %H:%M:%S'), (base_date - timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'), 'T5', '17:30:00', '20:30:00'),
                        ('KMR-1013', 'Scheduled', None, (base_date - timedelta(hours=21)).strftime('%Y-%m-%d %H:%M:%S'), (base_date - timedelta(hours=14)).strftime('%Y-%m-%d %H:%M:%S'), 'T4', '09:45:00', '11:45:00'),
                        ('KMR-1014', 'Pending', None, (base_date - timedelta(hours=13)).strftime('%Y-%m-%d %H:%M:%S'), (base_date - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'), 'T1', '15:45:00', '16:45:00'),
                        ('KMR-1015', 'Pending', None, (base_date - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'), (base_date + timedelta(hours=13)).strftime('%Y-%m-%d %H:%M:%S'), 'T3', '17:00:00', '20:00:00'),
                        ('KMR-1016', 'Sanitized', None, (base_date - timedelta(hours=13)).strftime('%Y-%m-%d %H:%M:%S'), (base_date - timedelta(hours=10)).strftime('%Y-%m-%d %H:%M:%S'), 'T2', '11:00:00', '15:00:00'),
                        ('KMR-1017', 'Scheduled', None, (base_date - timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S'), (base_date + timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S'), 'T3', '16:30:00', '18:30:00'),
                        ('KMR-1018', 'Scheduled', None, (base_date - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M:%S'), (base_date).strftime('%Y-%m-%d %H:%M:%S'), 'T2', '14:15:00', '17:15:00'),
                        ('KMR-1019', 'In Progress', None, (base_date - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M:%S'), (base_date - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'), 'T2', '11:00:00', '13:00:00'),
                        ('KMR-1020', 'Pending', None, (base_date - timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S'), (base_date - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'), 'T3', '06:00:00', '07:00:00'),
                        ('KMR-1021', 'Scheduled', None, (base_date - timedelta(hours=7)).strftime('%Y-%m-%d %H:%M:%S'), (base_date + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'), 'T3', '15:15:00', '16:15:00'),
                        ('KMR-1022', 'In Progress', None, (base_date - timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'), (base_date - timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S'), 'T2', '09:15:00', '10:15:00'),
                        ('KMR-1023', 'Sanitized', None, (base_date - timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S'), (base_date + timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S'), 'T1', '15:15:00', '18:15:00'),
                        ('KMR-1024', 'Pending', None, (base_date - timedelta(hours=16)).strftime('%Y-%m-%d %H:%M:%S'), (base_date - timedelta(hours=13)).strftime('%Y-%m-%d %H:%M:%S'), 'T1', '11:15:00', '12:15:00'),
                        ('KMR-1025', 'In Progress', None, (base_date - timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'), (base_date + timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S'), 'T5', '14:30:00', '17:30:00'),
                    ]
                    
                    for data in sample_data:
                        try:
                            cursor.execute('''
                                INSERT INTO cleaning 
                                (train_id, status_record, computed_status, last_sanitized, next_schedule, team_id, time_slot_start, time_slot_end)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (train_id) DO NOTHING
                            ''', data)
                        except Exception as e:
                            logger.error(f"Error inserting {data[0]}: {e}")
                    
                    logger.info("Comprehensive sample data inserted successfully")
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            logger.error(traceback.format_exc())
            return False
        finally:
            conn.close()

    def compute_status(self, record):
        """Compute status based on schedule and current time"""
        try:
            if record.get("status_record") == "Sanitized":
                return "Clean"
            elif record.get("status_record") == "In Progress":
                return "In Progress"
            elif record.get("next_schedule"):
                if isinstance(record["next_schedule"], str):
                    next_schedule = datetime.fromisoformat(record["next_schedule"].replace('Z', '+00:00'))
                else:
                    next_schedule = record["next_schedule"]
                
                now = datetime.now()
                if next_schedule < now:
                    return "Overdue"
                else:
                    return "Pending"
            else:
                return record.get("status_record", "Unknown")
        except Exception as e:
            logger.error(f"Error computing status: {e}")
            return record.get("status_record", "Unknown")

    def fetch_cleaning_data(self):
        """Fetch cleaning data from database"""
        conn = self.get_db_connection()
        if not conn:
            raise Exception("Database connection failed")
        
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("SELECT COUNT(*) FROM cleaning")
                total_count = cursor.fetchone()['count']
                logger.info(f"Total records in database: {total_count}")
                
                cursor.execute('''
                    SELECT train_id, status_record, computed_status, last_sanitized, 
                           next_schedule, team_id, time_slot_start, time_slot_end,
                           created_at, updated_at
                    FROM cleaning 
                    ORDER BY train_id
                ''')
                
                rows = cursor.fetchall()
                logger.info(f"Fetched {len(rows)} records from database")
                
                # Convert to list of dictionaries and format dates
                data = []
                for row in rows:
                    record = dict(row)
                    
                    # Format datetime fields
                    if record['last_sanitized']:
                        record['last_sanitized'] = record['last_sanitized'].isoformat() + 'Z'
                    if record['next_schedule']:
                        record['next_schedule'] = record['next_schedule'].isoformat() + 'Z'
                    if record['created_at']:
                        record['created_at'] = record['created_at'].isoformat() + 'Z'
                    if record['updated_at']:
                        record['updated_at'] = record['updated_at'].isoformat() + 'Z'
                    
                    # Format time fields
                    if record['time_slot_start']:
                        record['time_slot_start'] = str(record['time_slot_start'])[:5]
                    if record['time_slot_end']:
                        record['time_slot_end'] = str(record['time_slot_end'])[:5]
                        
                    # Compute status if not present
                    if not record.get("computed_status"):
                        record["computed_status"] = self.compute_status(record)
                    
                    data.append(record)
                
                logger.info(f"Returning {len(data)} formatted records to frontend")
                return data
                
        except Exception as e:
            logger.error(f"Database query error: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            conn.close()

    def calculate_summary_statistics(self, rows):
        """Calculate summary statistics from cleaning data"""
        current_date = datetime.now()
        today_start = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        cleaned_today = 0
        total_cleaned = 0
        pending = 0
        in_progress = 0
        overdue = 0
        scheduled = 0
        
        for row in rows:
            status = row.get("computed_status") or self.compute_status(row)
            status_record = row.get("status_record", "")
            
            if status == "Clean" or status_record == "Sanitized":
                total_cleaned += 1
                if row.get("last_sanitized"):
                    try:
                        last_sanitized = row["last_sanitized"]
                        if isinstance(last_sanitized, str):
                            last_sanitized = datetime.fromisoformat(last_sanitized.replace('Z', '+00:00'))
                        if last_sanitized >= today_start:
                            cleaned_today += 1
                    except:
                        pass
            elif status == "Pending" or status_record == "Pending":
                pending += 1
            elif status == "In Progress" or status_record == "In Progress":
                in_progress += 1
            elif status == "Overdue":
                overdue += 1
            elif status_record == "Scheduled":
                scheduled += 1
        
        return {
            'cleaned_today': cleaned_today,
            'cleaned': total_cleaned,
            'pending': pending,
            'in_progress': in_progress,
            'overdue': overdue,
            'scheduled': scheduled
        }

    def generate_team_data(self, rows):
        """Generate team data with realistic status based on current assignments"""
        active_teams = {}
        for row in rows:
            if row.get('team_id') and row.get('status_record') == 'In Progress':
                active_teams[row['team_id']] = row['train_id']
        
        teams = [
            {
                "team_id": "Team A (T1)", 
                "status": "Busy" if 'T1' in active_teams else "Available", 
                "assigned": "3 members", 
                "current_task": active_teams.get('T1', 'None')
            },
            {
                "team_id": "Team B (T2)", 
                "status": "Busy" if 'T2' in active_teams else "Available", 
                "assigned": "2 members", 
                "current_task": active_teams.get('T2', 'None')
            },
            {
                "team_id": "Team C (T3)", 
                "status": "Busy" if 'T3' in active_teams else "Available", 
                "assigned": "4 members", 
                "current_task": active_teams.get('T3', 'None')
            },
            {
                "team_id": "Team D (T4)", 
                "status": "Busy" if 'T4' in active_teams else "Available", 
                "assigned": "3 members", 
                "current_task": active_teams.get('T4', 'None')
            },
            {
                "team_id": "Team E (T5)", 
                "status": "Busy" if 'T5' in active_teams else "Available", 
                "assigned": "2 members", 
                "current_task": active_teams.get('T5', 'None')
            }
        ]
        
        return teams

    def generate_alerts(self, stats, teams):
        """Generate alerts based on actual database data"""
        alerts = []
        
        if stats['overdue'] > 0:
            alerts.append(f"{stats['overdue']} train(s) overdue for sanitization")
        if stats['in_progress'] > 0:
            alerts.append(f"{stats['in_progress']} train(s) currently being sanitized")
        
        # Add database status
        alerts.append("Database connected - live data active")
        
        # Add operational alerts based on data
        if stats['cleaned_today'] == 0:
            alerts.append("No trains cleaned today yet")
        elif stats['cleaned_today'] > 10:
            alerts.append(f"High activity: {stats['cleaned_today']} trains cleaned today")
            
        busy_teams = len([t for t in teams if t['status'] == 'Busy'])
        if busy_teams == 0:
            alerts.append("All teams available for assignment")
        elif busy_teams >= 4:
            alerts.append(f"Resource alert: {busy_teams} teams currently busy")
        
        return alerts

    def update_train_status(self, train_id, new_status, team_id=None):
        """Update cleaning status for a specific train"""
        conn = self.get_db_connection()
        if not conn:
            raise Exception("Database connection failed")
        
        try:
            with conn.cursor() as cursor:
                # Update the record
                update_query = """
                    UPDATE cleaning 
                    SET status_record = %s, 
                        computed_status = %s,
                        team_id = COALESCE(%s, team_id),
                        updated_at = CURRENT_TIMESTAMP
                """
                
                params = [new_status, new_status, team_id]
                
                # If marking as sanitized, update last_sanitized
                if new_status == "Sanitized":
                    update_query += ", last_sanitized = CURRENT_TIMESTAMP"
                
                update_query += " WHERE train_id = %s"
                params.append(train_id)
                
                cursor.execute(update_query, params)
                
                if cursor.rowcount == 0:
                    raise Exception(f"Train ID {train_id} not found")
                
                conn.commit()
                logger.info(f"Updated status for {train_id} to {new_status} (Team: {team_id})")
                
                return {"message": f"Status updated for {train_id}", "status": "success"}
                
        except Exception as e:
            logger.error(f"Update error: {e}")
            raise
        finally:
            conn.close()

    def get_debug_info(self):
        """Debug endpoint to check database state"""
        conn = self.get_db_connection()
        if not conn:
            raise Exception("Database connection failed")
        
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                # Get table info
                cursor.execute("SELECT COUNT(*) as total FROM cleaning")
                total = cursor.fetchone()['total']
                
                cursor.execute("SELECT train_id, status_record FROM cleaning ORDER BY train_id LIMIT 10")
                sample_records = cursor.fetchall()
                
                cursor.execute("SELECT status_record, COUNT(*) as count FROM cleaning GROUP BY status_record")
                status_counts = cursor.fetchall()
                
                return {
                    "total_records": total,
                    "sample_records": [dict(r) for r in sample_records],
                    "status_distribution": [dict(r) for r in status_counts],
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Debug query error: {e}")
            raise
        finally:
            conn.close()

# Initialize cleaning manager
cleaning_manager = CleaningManager()

# Routes
@app.route("/", methods=["GET"])
def dashboard():
    return send_from_directory('.', 'cleaning.html')

@app.route("/status", methods=["GET"])
def status():
    connectivity_ok = cleaning_manager.test_connectivity()
    db_connected = cleaning_manager.get_db_connection() is not None
    
    return jsonify({
        "message": "Kochi Metro Cleaning Backend is running",
        "network_connectivity": connectivity_ok,
        "database_connected": db_connected,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/cleaning", methods=["GET"])
def get_cleaning_data():
    """Get cleaning data from database"""
    try:
        data = cleaning_manager.fetch_cleaning_data()
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"API error in /api/cleaning: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/cleaning/summary", methods=["GET"])
def get_summary():
    """Get summary data from database"""
    try:
        rows = cleaning_manager.fetch_cleaning_data()
        stats = cleaning_manager.calculate_summary_statistics(rows)
        teams = cleaning_manager.generate_team_data(rows)
        alerts = cleaning_manager.generate_alerts(stats, teams)
        
        return jsonify({
            **stats,
            "teams": teams,
            "alerts": alerts
        })
        
    except Exception as e:
        logger.error(f"API error in /api/cleaning/summary: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/cleaning/update", methods=["POST"])
def update_cleaning_status():
    """Update cleaning status for a specific train"""
    try:
        data = request.json
        train_id = data.get('train_id')
        new_status = data.get('status')
        team_id = data.get('team_id')
        
        if not train_id or not new_status:
            return jsonify({"error": "train_id and status are required"}), 400
        
        result = cleaning_manager.update_train_status(train_id, new_status, team_id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error in /api/cleaning/update: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/debug/cleaning", methods=["GET"])
def debug_cleaning_data():
    """Debug endpoint to check database state"""
    try:
        debug_info = cleaning_manager.get_debug_info()
        return jsonify(debug_info)
        
    except Exception as e:
        logger.error(f"API error in /debug/cleaning: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint with database validation"""
    try:
        connectivity_ok = cleaning_manager.test_connectivity()
        db_connected = cleaning_manager.get_db_connection() is not None
        
        # Test actual database operations
        record_count = 0
        if db_connected:
            try:
                conn = cleaning_manager.get_db_connection()
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM cleaning")
                    record_count = cur.fetchone()[0]
                conn.close()
            except:
                db_connected = False
        
        return jsonify({
            "status": "healthy" if (connectivity_ok and db_connected) else "degraded",
            "service": "Kochi Metro Cleaning API",
            "network": "ok" if connectivity_ok else "failed",
            "database": "connected" if db_connected else "disconnected",
            "record_count": record_count,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

if __name__ == "__main__":
    print("Starting Kochi Metro Cleaning Dashboard...")
    print("Dashboard will be available at: http://127.0.0.1:5003")
    print("API endpoints available at: http://127.0.0.1:5003/api/")
    print()
    
    # Test connectivity first
    print("Testing network connectivity...")
    if cleaning_manager.test_connectivity():
        print("✓ Network connectivity: OK")
    else:
        print("✗ Network connectivity: FAILED")
        print("This may be due to:")
        print("1. No internet connection")
        print("2. Firewall blocking the connection")
        print("3. DNS resolution issues")
        print("4. Supabase service temporarily unavailable")
    
    # Test database
    print("\nTesting database connection...")
    if cleaning_manager.get_db_connection():
        print("✓ Database connected and initialized successfully")
        print("✓ Application will use live database data only")
    else:
        print("✗ Database connection failed")
        print("❌ Application will return database errors")
    
    app.run(debug=True, port=5003, host='127.0.0.1')