from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mock model statistics (replace with actual model data)
MODEL_STATS = [
    {'label': 'Accuracy', 'value': '92.5%'},
    {'label': 'Precision', 'value': '89.3%'},
    {'label': 'Recall', 'value': '91.8%'},
    {'label': 'F1 Score', 'value': '90.5%'},
    {'label': 'Training Time', 'value': '45 min'},
    {'label': 'Dataset Size', 'value': '10,000 samples'}
]

# Navigation items
NAVIGATION_ITEMS = [
    {'name': 'Home', 'url': '#home'},
    {'name': 'Analyze Data', 'url': '#analyze'},
    {'name': 'Upload Data', 'url': '#upload'},
    {'name': 'View Results', 'url': '#results'},
    {'name': 'Model Stats', 'url': '#stats'},
    {'name': 'Tune Model', 'url': '#tune'},
    {'name': 'About', 'url': '#about'}
]

# Model types for hyperparameter tuning
MODEL_TYPES = [
    {'value': 'random_forest', 'label': 'Random Forest'},
    {'value': 'gradient_boosting', 'label': 'Gradient Boosting'},
    {'value': 'svm', 'label': 'Support Vector Machine'},
    {'value': 'neural_network', 'label': 'Neural Network'}
]

# Sample chart data
CHART_DATA = {
    'datasets': [
        {
            'label': 'Confirmed Exoplanets',
            'data': [
                {'x': 10.5, 'y': 1.2},
                {'x': 50.3, 'y': 2.5},
                {'x': 100.7, 'y': 0.8},
                {'x': 20.1, 'y': 1.5},
                {'x': 80.4, 'y': 3.0}
            ],
            'backgroundColor': '#a855f7',
            'borderColor': '#a855f7',
            'pointRadius': 6
        },
        {
            'label': 'Planetary Candidates',
            'data': [
                {'x': 15.2, 'y': 1.8},
                {'x': 60.9, 'y': 2.0},
                {'x': 120.3, 'y': 1.0},
                {'x': 30.5, 'y': 1.7}
            ],
            'backgroundColor': '#3b82f6',
            'borderColor': '#3b82f6',
            'pointRadius': 6
        },
        {
            'label': 'False Positives',
            'data': [
                {'x': 25.6, 'y': 2.2},
                {'x': 70.1, 'y': 1.9},
                {'x': 90.8, 'y': 1.3}
            ],
            'backgroundColor': '#ef4444',
            'borderColor': '#ef4444',
            'pointRadius': 6
        }
    ]
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html',
        page_title='Exoplanet Data Explorer',
        site_name='Exoplanet Explorer',
        navigation_items=NAVIGATION_ITEMS,
        hero_title='Discover Exoplanets with AI',
        hero_description='Explore NASA\'s exoplanet datasets, classify potential exoplanets, and visualize results with our AI-powered model. Upload data, tune hyperparameters, or view model performance metrics.',
        model_stats=MODEL_STATS,
        model_types=MODEL_TYPES,
        chart_data=CHART_DATA,
        current_year=datetime.now().year,
        footer_text='Exoplanet Explorer. Powered by xAI.',
        x_axis_label='Orbital Period (days)',
        y_axis_label='Planetary Radius (Earth radii)'
    )


@app.route('/search', methods=['GET'])
def search():
    """Handle search requests"""
    query = request.args.get('query', '')
    
    # TODO: Implement actual search logic
    # For now, just display the search query
    
    # Mock search results - filter chart data based on query
    search_results = f"Search results for: {query}"
    
    return render_template('index.html',
        page_title=f'Search Results - {query}',
        site_name='Exoplanet Explorer',
        navigation_items=NAVIGATION_ITEMS,
        search_query=query,
        model_stats=MODEL_STATS,
        model_types=MODEL_TYPES,
        chart_data=CHART_DATA,
        current_year=datetime.now().year,
        analysis_result=search_results
    )


@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle data analysis requests"""
    # Get form data
    orbital_period = request.form.get('orbital_period')
    transit_duration = request.form.get('transit_duration')
    planetary_radius = request.form.get('planetary_radius')
    stellar_magnitude = request.form.get('stellar_magnitude')
    
    # Validate inputs
    if not all([orbital_period, transit_duration, planetary_radius, stellar_magnitude]):
        flash('Please fill in all fields', 'error')
        return redirect(url_for('index') + '#analyze')
    
    # TODO: Implement actual analysis logic with ML model
    # Mock analysis result
    try:
        op = float(orbital_period)
        td = float(transit_duration)
        pr = float(planetary_radius)
        sm = float(stellar_magnitude)
        
        # Simple mock classification
        if pr > 1.5 and op > 50:
            classification = "Likely Exoplanet Candidate"
            confidence = "85%"
        elif pr > 0.8 and op > 20:
            classification = "Possible Exoplanet"
            confidence = "65%"
        else:
            classification = "Unlikely to be an Exoplanet"
            confidence = "40%"
        
        analysis_result = f"Classification: {classification} (Confidence: {confidence}). Orbital Period: {op} days, Transit Duration: {td} hours, Planetary Radius: {pr} Earth radii, Stellar Magnitude: {sm}"
        
        # Store form data to repopulate form
        form_data = {
            'orbital_period': orbital_period,
            'transit_duration': transit_duration,
            'planetary_radius': planetary_radius,
            'stellar_magnitude': stellar_magnitude
        }
        
        return render_template('index.html',
            page_title='Analysis Results',
            site_name='Exoplanet Explorer',
            navigation_items=NAVIGATION_ITEMS,
            model_stats=MODEL_STATS,
            model_types=MODEL_TYPES,
            chart_data=CHART_DATA,
            current_year=datetime.now().year,
            form_data=form_data,
            analysis_result=analysis_result
        )
    
    except ValueError:
        flash('Please enter valid numeric values', 'error')
        return redirect(url_for('index') + '#analyze')


@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload requests"""
    # Check if file is present
    if 'dataset' not in request.files:
        return render_template('index.html',
            page_title='Upload Error',
            site_name='Exoplanet Explorer',
            navigation_items=NAVIGATION_ITEMS,
            model_stats=MODEL_STATS,
            model_types=MODEL_TYPES,
            chart_data=CHART_DATA,
            current_year=datetime.now().year,
            upload_message='No file selected',
            upload_success=False
        )
    
    file = request.files['dataset']
    
    # Check if filename is empty
    if file.filename == '':
        return render_template('index.html',
            page_title='Upload Error',
            site_name='Exoplanet Explorer',
            navigation_items=NAVIGATION_ITEMS,
            model_stats=MODEL_STATS,
            model_types=MODEL_TYPES,
            chart_data=CHART_DATA,
            current_year=datetime.now().year,
            upload_message='No file selected',
            upload_success=False
        )
    
    # Check if file is allowed
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # TODO: Process the uploaded file with your model
        
        return render_template('index.html',
            page_title='Upload Success',
            site_name='Exoplanet Explorer',
            navigation_items=NAVIGATION_ITEMS,
            model_stats=MODEL_STATS,
            model_types=MODEL_TYPES,
            chart_data=CHART_DATA,
            current_year=datetime.now().year,
            upload_message=f'File "{filename}" uploaded successfully! Processing data...',
            upload_success=True
        )
    else:
        return render_template('index.html',
            page_title='Upload Error',
            site_name='Exoplanet Explorer',
            navigation_items=NAVIGATION_ITEMS,
            model_stats=MODEL_STATS,
            model_types=MODEL_TYPES,
            chart_data=CHART_DATA,
            current_year=datetime.now().year,
            upload_message='Invalid file type. Please upload a CSV or TXT file.',
            upload_success=False
        )


@app.route('/tune', methods=['POST'])
def tune():
    """Handle hyperparameter tuning requests"""
    # Get hyperparameters from form
    learning_rate = request.form.get('learning_rate')
    model_type = request.form.get('model_type')
    max_depth = request.form.get('max_depth')
    min_samples_split = request.form.get('min_samples_split')
    
    # Store hyperparameters
    hyperparams = {
        'learning_rate': learning_rate,
        'model_type': model_type,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split
    }
    
    # TODO: Implement actual model retraining with new hyperparameters
    
    tune_message = f"Model updated successfully with {model_type}!"
    if learning_rate:
        tune_message += f" Learning Rate: {learning_rate}"
    if max_depth:
        tune_message += f", Max Depth: {max_depth}"
    if min_samples_split:
        tune_message += f", Min Samples Split: {min_samples_split}"
    
    return render_template('index.html',
        page_title='Model Tuned',
        site_name='Exoplanet Explorer',
        navigation_items=NAVIGATION_ITEMS,
        model_stats=MODEL_STATS,
        model_types=MODEL_TYPES,
        chart_data=CHART_DATA,
        current_year=datetime.now().year,
        hyperparams=hyperparams,
        tune_message=tune_message
    )


@app.route('/api/chart-data')
def get_chart_data():
    """API endpoint to get chart data dynamically"""
    return jsonify(CHART_DATA)


@app.route('/api/stats')
def get_stats():
    """API endpoint to get model statistics"""
    return jsonify(MODEL_STATS)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('index.html',
        page_title='Page Not Found',
        site_name='Exoplanet Explorer',
        navigation_items=NAVIGATION_ITEMS,
        model_stats=MODEL_STATS,
        model_types=MODEL_TYPES,
        chart_data=CHART_DATA,
        current_year=datetime.now().year,
        analysis_result='Page not found. Please use the navigation menu.'
    ), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('index.html',
        page_title='Server Error',
        site_name='Exoplanet Explorer',
        navigation_items=NAVIGATION_ITEMS,
        model_stats=MODEL_STATS,
        model_types=MODEL_TYPES,
        chart_data=CHART_DATA,
        current_year=datetime.now().year,
        analysis_result='An internal error occurred. Please try again later.'
    ), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
