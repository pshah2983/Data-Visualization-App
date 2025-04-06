from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from werkzeug.utils import secure_filename
import pandas as pd
import json
from datetime import datetime, timedelta
import pytz
import io
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-GUI
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
from flask_session import Session

# Create IST timezone object
IST = pytz.timezone('Asia/Kolkata')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Use a fixed secret key instead of random
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///new_users.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Session configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
Session(app)  # Initialize Flask-Session

# Initialize other extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.session_protection = "strong"

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    datasets = db.relationship('Dataset', backref='user', lazy=True)
    visualizations = db.relationship('Visualization', backref='user', lazy=True)
    last_login = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(IST))

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(IST))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)
    columns = db.Column(db.Text)  # Store column names as JSON string

class Visualization(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    chart_type = db.Column(db.String(50), nullable=False)
    x_axis = db.Column(db.String(100))
    y_axis = db.Column(db.String(100))
    color_by = db.Column(db.String(100))
    created_date = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(IST))
    config = db.Column(db.Text)  # Store additional configuration as JSON string

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=True)  # Enable remember me
            # Update last_login with IST time
            user.last_login = datetime.now(IST)
            db.session.commit()
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return redirect(url_for('register'))
        if len(password) < 7:
            flash('Password must be at least 7 characters long')
            return redirect(url_for('register'))
        user = User(username=username, password_hash=generate_password_hash(password), email=email)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    datasets = Dataset.query.filter_by(user_id=current_user.id).all()
    visualizations = Visualization.query.filter_by(user_id=current_user.id).all()
    
    # Format last_login in IST with custom format
    last_login = 'Never'
    if current_user.last_login:
        # Convert UTC to IST
        ist_time = current_user.last_login.astimezone(IST)
        last_login = ist_time.strftime('%d/%m/%Y %I:%M %p')
    
    return render_template('dashboard.html', 
                          username=current_user.username,
                          datasets=datasets, 
                          visualizations=visualizations,
                          last_login=last_login)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method != 'POST':
        return redirect(url_for('dashboard'))
    
    try:
        if 'file' not in request.files:
            flash('No file selected. Please choose a file to upload.', 'error')
            return redirect(url_for('dashboard'))
        
        file = request.files['file']
        dataset_name = request.form.get('dataset_name', '').strip()
        
        if not dataset_name:
            flash('Please provide a name for your dataset.', 'error')
            return redirect(url_for('dashboard'))
        
        if file.filename == '':
            flash('No file selected. Please choose a file to upload.', 'error')
            return redirect(url_for('dashboard'))
        
        if not allowed_file(file.filename):
            flash('File type not allowed. Please upload a CSV, Excel, or TXT file.', 'error')
            return redirect(url_for('dashboard'))
        
        # Create upload folder if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save file with unique name
        filename = secure_filename(file.filename)
        unique_filename = f"{current_user.id}_{int(datetime.utcnow().timestamp())}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        try:
            file.save(file_path)
            
            # Read the file to get column information
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif filename.endswith('.txt'):
                df = pd.read_csv(file_path, sep='\t')
            else:
                raise ValueError("Unsupported file format")
            
            # Store column names as JSON
            columns_json = json.dumps(df.columns.tolist())
            
            # Create dataset record
            dataset = Dataset(
                name=dataset_name,
                filename=unique_filename,
                user_id=current_user.id,
                file_type=filename.rsplit('.', 1)[1].lower(),
                columns=columns_json
            )
            db.session.add(dataset)
            db.session.commit()
            
            flash(f'Dataset "{dataset_name}" uploaded successfully!', 'success')
            return redirect(url_for('preprocess', dataset_id=dataset.id))
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            flash(f'Error processing file: {str(e)}', 'error')
            
    except Exception as e:
        flash(f'An unexpected error occurred: {str(e)}', 'error')
    
    return redirect(url_for('dashboard'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xlsx', 'txt'}

@app.route('/preprocess/<int:dataset_id>', methods=['GET', 'POST'])
@login_required
def preprocess(dataset_id):
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        if dataset.user_id != current_user.id:
            flash('You do not have permission to access this dataset', 'error')
            return redirect(url_for('dashboard'))
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
        
        try:
            # Read the dataset
            if dataset.file_type == 'csv':
                df = pd.read_csv(file_path)
            elif dataset.file_type == 'xlsx':
                df = pd.read_excel(file_path)
            elif dataset.file_type == 'txt':
                df = pd.read_csv(file_path, sep='\t')
            
            # Basic dataset information
            info = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'missing_cells': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum()
            }
            
            # Column information
            columns_info = []
            for column in df.columns:
                col_info = {
                    'name': column,
                    'dtype': str(df[column].dtype),
                    'missing_values': int(df[column].isnull().sum()),
                    'unique_values': int(df[column].nunique()),
                    'sample_values': df[column].dropna().head(3).tolist()
                }
                columns_info.append(col_info)
            
            if request.method == 'POST':
                # Get selected operations from form
                drop_duplicates = request.form.get('drop_duplicates') == 'on'
                drop_na_rows = request.form.get('drop_na_rows') == 'on'
                selected_columns = request.form.getlist('selected_columns')
                
                # Create a copy of the dataframe
                cleaned_df = df.copy()
                
                # Apply selected operations
                if drop_duplicates:
                    cleaned_df = cleaned_df.drop_duplicates()
                
                if drop_na_rows:
                    cleaned_df = cleaned_df.dropna()
                
                # Keep only selected columns if any are selected
                if selected_columns:
                    cleaned_df = cleaned_df[selected_columns]
                
                # Save the cleaned dataset
                cleaned_filename = f"cleaned_{dataset.filename}"
                cleaned_path = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)
                
                if dataset.file_type == 'csv':
                    cleaned_df.to_csv(cleaned_path, index=False)
                elif dataset.file_type == 'xlsx':
                    cleaned_df.to_excel(cleaned_path, index=False)
                elif dataset.file_type == 'txt':
                    cleaned_df.to_csv(cleaned_path, sep='\t', index=False)
                
                # Update dataset record
                dataset.filename = cleaned_filename
                dataset.columns = json.dumps(cleaned_df.columns.tolist())
                db.session.commit()
                
                flash('Data cleaning completed successfully!', 'success')
                return redirect(url_for('visualize', dataset_id=dataset.id))
            
            # For GET request, show the data preview and cleaning options
            preview_data = df.head(5).to_dict('records')
            
            return render_template('preprocess.html',
                                dataset=dataset,
                                info=info,
                                columns_info=columns_info,
                                preview_data=preview_data,
                                columns=df.columns.tolist())
            
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
            flash(f'Error processing dataset: {str(e)}', 'error')
            return redirect(url_for('dashboard'))
            
    except Exception as e:
        print(f"Error accessing dataset: {str(e)}")
        flash(f'Error accessing dataset: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/visualize/<int:dataset_id>', methods=['GET', 'POST'])
@login_required
def visualize(dataset_id):
    try:
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # Check if the dataset belongs to the current user
        if dataset.user_id != current_user.id:
            flash('You do not have permission to access this dataset', 'error')
            return redirect(url_for('dashboard'))
        
        # Load the dataset
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
        
        try:
            # Read the dataset
            if dataset.file_type == 'csv':
                df = pd.read_csv(file_path)
            elif dataset.file_type == 'xlsx':
                df = pd.read_excel(file_path)
            elif dataset.file_type == 'txt':
                df = pd.read_csv(file_path, sep='\t')
            
            # Get column types for better selection options
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Create a dictionary of column types
            columns = {
                'numeric': numeric_columns,
                'categorical': categorical_columns,
                'datetime': datetime_columns,
                'all': df.columns.tolist()
            }
            
            if request.method == 'POST':
                viz_name = request.form.get('viz_name', '').strip()
                chart_type = request.form.get('chart_type')
                x_axis = request.form.get('x_axis')
                y_axis = request.form.get('y_axis')
                color_by = request.form.get('color_by')
                
                # Validate inputs
                if not viz_name:
                    flash('Please provide a name for your visualization', 'error')
                    return render_template('visualize.html', 
                                        dataset=dataset, 
                                        columns=columns, 
                                        sample_data=df.head(5).to_dict('records'))
                
                if not x_axis or (not y_axis and chart_type != 'pie'):
                    flash('Please select both X and Y axis columns (except for pie charts)', 'error')
                    return render_template('visualize.html', 
                                        dataset=dataset, 
                                        columns=columns, 
                                        sample_data=df.head(5).to_dict('records'))
                
                # Create visualization record
                viz = Visualization(
                    name=viz_name,
                    dataset_id=dataset_id,
                    user_id=current_user.id,
                    chart_type=chart_type,
                    x_axis=x_axis,
                    y_axis=y_axis,
                    color_by=color_by if color_by != 'None' else None,
                    created_date=datetime.now(IST)
                )
                db.session.add(viz)
                db.session.commit()
                
                flash(f'Visualization "{viz_name}" created successfully!', 'success')
                return redirect(url_for('view_visualization', viz_id=viz.id))
            
            # For GET request, show the visualization form
            sample_data = df.head(5).to_dict('records')
            return render_template('visualize.html', 
                                dataset=dataset, 
                                columns=columns, 
                                sample_data=sample_data)
            
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
            flash(f'Error processing dataset: {str(e)}', 'error')
            return redirect(url_for('dashboard'))
            
    except Exception as e:
        print(f"Error accessing dataset: {str(e)}")
        flash(f'Error accessing dataset: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/view/<int:viz_id>')
@login_required
def view_visualization(viz_id):
    try:
        viz = Visualization.query.get_or_404(viz_id)
        
        # Check if the visualization belongs to the current user
        if viz.user_id != current_user.id:
            flash('You do not have permission to access this visualization')
            return redirect(url_for('dashboard'))
        
        # Load the dataset
        dataset = Dataset.query.get(viz.dataset_id)
        if not dataset:
            flash('Dataset not found')
            return redirect(url_for('dashboard'))
            
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
        if not os.path.exists(file_path):
            flash('Dataset file not found')
            return redirect(url_for('dashboard'))
        
        # Read the dataset
        try:
            if dataset.file_type == 'csv':
                df = pd.read_csv(file_path)
            elif dataset.file_type == 'xlsx':
                df = pd.read_excel(file_path)
            elif dataset.file_type == 'txt':
                df = pd.read_csv(file_path, sep='\t')
                
            # Validate required columns exist
            required_columns = [viz.x_axis]
            if viz.y_axis:  # y_axis might be optional for pie charts
                required_columns.append(viz.y_axis)
            if viz.color_by:
                required_columns.append(viz.color_by)
                
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Identify numeric and categorical columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            # Handle data based on chart type and column types
            if viz.y_axis in df.columns:
                # Convert Y-axis to numeric if it's not already
                if viz.y_axis not in numeric_cols:
                    df[viz.y_axis] = pd.to_numeric(df[viz.y_axis], errors='coerce')
            
            if viz.x_axis in df.columns:
                # Convert X-axis to numeric only if needed (line, scatter, histogram)
                if viz.chart_type in ['line', 'scatter', 'histogram'] and viz.x_axis not in numeric_cols:
                    df[viz.x_axis] = pd.to_numeric(df[viz.x_axis], errors='coerce')
            
            # Drop rows with missing values in required columns
            df = df.dropna(subset=[col for col in required_columns if col in df.columns])
            
            if len(df) == 0:
                raise ValueError("No valid data points after cleaning")
            
            # Create the visualization
            plt.figure(figsize=(10, 6))
            plt.style.use('default')  # Use default style instead of seaborn
            
            if viz.chart_type == 'line':
                if viz.color_by:
                    for group in df[viz.color_by].unique():
                        group_data = df[df[viz.color_by] == group]
                        plt.plot(group_data[viz.x_axis], group_data[viz.y_axis], 
                               label=str(group), marker='o')
                    plt.legend(title=viz.color_by)
                else:
                    plt.plot(df[viz.x_axis], df[viz.y_axis], marker='o')
            
            elif viz.chart_type == 'bar':
                if viz.color_by:
                    grouped_data = df.groupby([viz.x_axis, viz.color_by])[viz.y_axis].sum().unstack()
                    grouped_data.plot(kind='bar', ax=plt.gca())
                    plt.legend(title=viz.color_by)
                else:
                    # Sort values by Y-axis for better visualization
                    df_sorted = df.sort_values(by=viz.y_axis, ascending=True)
                    plt.bar(df_sorted[viz.x_axis].astype(str), df_sorted[viz.y_axis])
            
            elif viz.chart_type == 'scatter':
                if viz.color_by:
                    scatter = plt.scatter(df[viz.x_axis], df[viz.y_axis], 
                                       c=pd.factorize(df[viz.color_by])[0], 
                                       cmap='viridis', alpha=0.6)
                    plt.colorbar(scatter, label=viz.color_by)
                    # Add legend for categorical color_by
                    handles = [plt.scatter([], [], c=c, label=l) 
                              for c, l in zip(plt.cm.viridis(np.linspace(0, 1, len(df[viz.color_by].unique()))),
                                            df[viz.color_by].unique())]
                    plt.legend(handles=handles, title=viz.color_by)
                else:
                    plt.scatter(df[viz.x_axis], df[viz.y_axis], alpha=0.6)
            
            elif viz.chart_type == 'pie':
                if viz.color_by:
                    pie_data = df.groupby(viz.color_by)[viz.x_axis].count()
                    plt.pie(pie_data, labels=pie_data.index.astype(str), autopct='%1.1f%%')
                else:
                    pie_data = df.groupby(viz.x_axis)[viz.x_axis].count()
                    plt.pie(pie_data, labels=pie_data.index.astype(str), autopct='%1.1f%%')
            
            elif viz.chart_type == 'histogram':
                plt.hist(df[viz.x_axis].astype(float), bins=30, edgecolor='black')
            
            plt.title(viz.name, pad=20, fontsize=14, fontweight='bold')
            if viz.chart_type != 'pie':
                plt.xlabel(viz.x_axis, fontsize=12)
                plt.ylabel(viz.y_axis if viz.y_axis else 'Count', fontsize=12)
                plt.xticks(rotation=45, ha='right')
            
            # Add grid for certain chart types
            if viz.chart_type in ['line', 'scatter']:
                plt.grid(True, linestyle='--', alpha=0.7)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Convert plot to base64 string
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            
            return render_template('view_visualization.html', 
                                  viz=viz, 
                                  dataset=dataset, 
                                  plot_url=plot_url)
        
        except Exception as e:
            print(f"Data processing error: {str(e)}")  # Debug log
            flash(f'Error processing data: {str(e)}')
            return redirect(url_for('dashboard'))
    
    except Exception as e:
        print(f"Visualization error: {str(e)}")  # Debug log
        flash(f'Error creating visualization: {str(e)}')
        return redirect(url_for('dashboard'))

@app.route('/edit/<int:viz_id>', methods=['GET', 'POST'])
@login_required
def edit_visualization(viz_id):
    viz = Visualization.query.get_or_404(viz_id)
    
    # Check if the visualization belongs to the current user
    if viz.user_id != current_user.id:
        flash('You do not have permission to edit this visualization')
        return redirect(url_for('dashboard'))
    
    # Load the dataset
    dataset = Dataset.query.get(viz.dataset_id)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
    
    try:
        if dataset.file_type == 'csv':
            df = pd.read_csv(file_path)
        elif dataset.file_type == 'xlsx':
            df = pd.read_excel(file_path)
        elif dataset.file_type == 'txt':
            df = pd.read_csv(file_path, sep='\t')
        
        columns = df.columns.tolist()
        
        if request.method == 'POST':
            # Update visualization parameters
            viz.name = request.form.get('viz_name', viz.name)
            viz.chart_type = request.form.get('chart_type', viz.chart_type)
            viz.x_axis = request.form.get('x_axis', viz.x_axis)
            viz.y_axis = request.form.get('y_axis', viz.y_axis)
            viz.color_by = request.form.get('color_by', viz.color_by)
            
            db.session.commit()
            
            flash(f'Visualization "{viz.name}" updated successfully!')
            return redirect(url_for('view_visualization', viz_id=viz.id))
        
        return render_template('edit_visualization.html', 
                              viz=viz, 
                              dataset=dataset, 
                              columns=columns)
    
    except Exception as e:
        flash(f'Error editing visualization: {str(e)}')
        return redirect(url_for('dashboard'))

@app.route('/delete/<int:dataset_id>')
@login_required
def delete_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check if the dataset belongs to the current user
    if dataset.user_id != current_user.id:
        flash('You do not have permission to delete this dataset')
        return redirect(url_for('dashboard'))
    
    # Delete the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Delete associated visualizations
    Visualization.query.filter_by(dataset_id=dataset_id).delete()
    
    # Delete the dataset record
    db.session.delete(dataset)
    db.session.commit()
    
    flash(f'Dataset "{dataset.name}" deleted successfully!')
    return redirect(url_for('dashboard'))

@app.route('/delete-viz/<int:viz_id>')
@login_required
def delete_visualization(viz_id):
    viz = Visualization.query.get_or_404(viz_id)
    
    # Check if the visualization belongs to the current user
    if viz.user_id != current_user.id:
        flash('You do not have permission to delete this visualization')
        return redirect(url_for('dashboard'))
    
    # Delete the visualization record
    db.session.delete(viz)
    db.session.commit()
    
    flash(f'Visualization "{viz.name}" deleted successfully!')
    return redirect(url_for('dashboard'))

@app.route('/download/<int:dataset_id>')
@login_required
def download_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Check if the dataset belongs to the current user
    if dataset.user_id != current_user.id:
        flash('You do not have permission to download this dataset')
        return redirect(url_for('dashboard'))
    
    # Get the file path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash('File not found')
        return redirect(url_for('dashboard'))

@app.route('/export-viz/<int:viz_id>')
@login_required
def export_visualization(viz_id):
    viz = Visualization.query.get_or_404(viz_id)
    
    # Check if the visualization belongs to the current user
    if viz.user_id != current_user.id:
        flash('You do not have permission to export this visualization')
        return redirect(url_for('dashboard'))
    
    # Load the dataset
    dataset = Dataset.query.get(viz.dataset_id)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
    
    try:
        if dataset.file_type == 'csv':
            df = pd.read_csv(file_path)
        elif dataset.file_type == 'xlsx':
            df = pd.read_excel(file_path)
        elif dataset.file_type == 'txt':
            df = pd.read_csv(file_path, sep='\t')
        
        # Create the visualization
        plt.figure(figsize=(10, 6))
        
        if viz.chart_type == 'line':
            if viz.color_by:
                for group in df[viz.color_by].unique():
                    group_data = df[df[viz.color_by] == group]
                    plt.plot(group_data[viz.x_axis], group_data[viz.y_axis], label=group)
                plt.legend()
            else:
                plt.plot(df[viz.x_axis], df[viz.y_axis])
        
        elif viz.chart_type == 'bar':
            if viz.color_by:
                df.groupby([viz.x_axis, viz.color_by])[viz.y_axis].sum().unstack().plot(kind='bar')
                plt.legend()
            else:
                plt.bar(df[viz.x_axis], df[viz.y_axis])
        
        elif viz.chart_type == 'scatter':
            if viz.color_by:
                plt.scatter(df[viz.x_axis], df[viz.y_axis], c=df[viz.color_by], cmap='viridis')
                plt.colorbar(label=viz.color_by)
            else:
                plt.scatter(df[viz.x_axis], df[viz.y_axis])
        
        elif viz.chart_type == 'pie':
            if viz.color_by:
                df.groupby(viz.color_by)[viz.y_axis].sum().plot(kind='pie', autopct='%1.1f%%')
            else:
                df.groupby(viz.x_axis)[viz.y_axis].sum().plot(kind='pie', autopct='%1.1f%%')
        
        elif viz.chart_type == 'histogram':
            plt.hist(df[viz.x_axis], bins=30)
        
        plt.title(viz.name)
        plt.xlabel(viz.x_axis)
        if viz.chart_type != 'pie':
            plt.ylabel(viz.y_axis)
        
        # Save to BytesIO
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
        img.seek(0)
        plt.close()
        
        return send_file(
            img,
            mimetype='image/png',
            as_attachment=True,
            download_name=f"{viz.name}.png"
        )
    
    except Exception as e:
        flash(f'Error exporting visualization: {str(e)}')
        return redirect(url_for('dashboard'))

if __name__ == '__main__':
    with app.app_context():
        # Create tables if they don't exist
        db.create_all()
        
        # Check if last_login column exists in User table
        inspector = db.inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('user')]
        
        if 'last_login' not in columns:
            # Add last_login column to User table
            with db.engine.connect() as conn:
                conn.execute(db.text('ALTER TABLE user ADD COLUMN last_login DATETIME'))
                conn.commit()
            print("Added last_login column to User table")
            
    app.run(debug=True) 