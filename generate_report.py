"""Generate interactive HTML report of training results"""

import csv
import json

# Read CSV
data = {
    'epochs': [],
    'map50': [],
    'map95': [],
    'precision': [],
    'recall': []
}

with open(r'd:\Github\EV-PassengerDection-RL\results\coco_full\passenger_detection5\results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data['epochs'].append(int(float(row['epoch'])))
        data['map50'].append(float(row['metrics/mAP50(B)']))
        data['map95'].append(float(row['metrics/mAP50-95(B)']))
        data['precision'].append(float(row['metrics/precision(B)']))
        data['recall'].append(float(row['metrics/recall(B)']))

# Get latest metrics
latest_epoch = data['epochs'][-1]
latest_map50 = data['map50'][-1]
latest_map95 = data['map95'][-1]
latest_precision = data['precision'][-1]
latest_recall = data['recall'][-1]

# Calculate improvements from epoch 1 to latest
map50_improvement = ((latest_map50 - data['map50'][0]) / data['map50'][0] * 100)
recall_improvement = ((latest_recall - data['recall'][0]) / data['recall'][0] * 100)

html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Training Results - COCO Person Detection</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{ font-size: 32px; margin-bottom: 10px; }}
        .header p {{ font-size: 16px; opacity: 0.9; }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        
        .metric-card.good {{ border-left-color: #10b981; }}
        .metric-card.warning {{ border-left-color: #f59e0b; }}
        .metric-card.poor {{ border-left-color: #ef4444; }}
        
        .metric-label {{ font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 8px; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #333; margin-bottom: 5px; }}
        .metric-change {{ font-size: 13px; color: #10b981; }}
        .metric-change.negative {{ color: #ef4444; }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .chart-container h3 {{
            font-size: 16px;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .chart-wrapper {{ position: relative; height: 300px; }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            padding: 20px;
        }}
        
        .stat-item {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 6px;
            border-left: 3px solid #667eea;
        }}
        
        .stat-label {{ font-size: 12px; color: #666; margin-bottom: 5px; }}
        .stat-value {{ font-size: 20px; font-weight: bold; color: #333; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Training Results Summary</h1>
            <p>COCO Person Detection - YOLOv11m (Epoch {latest_epoch}/50)</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card good">
                <div class="metric-label">mAP@50</div>
                <div class="metric-value">{latest_map50:.3f}</div>
                <div class="metric-change">↑ {map50_improvement:+.1f}%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">mAP@50-95</div>
                <div class="metric-value">{latest_map95:.3f}</div>
                <div class="metric-change">↑ Improving</div>
            </div>
            
            <div class="metric-card good">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{latest_precision:.3f}</div>
                <div class="metric-change">✓ Good</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{latest_recall:.3f}</div>
                <div class="metric-change">↑ {recall_improvement:+.1f}%</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <h3>📈 mAP Progress</h3>
                <div class="chart-wrapper">
                    <canvas id="mapChart"></canvas>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>🎯 Precision & Recall</h3>
                <div class="chart-wrapper">
                    <canvas id="prChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>📊 All Metrics Comparison</h3>
            <div class="chart-wrapper" style="height: 350px;">
                <canvas id="allChart"></canvas>
            </div>
        </div>
        
        <div class="chart-container">
            <h3>📌 Statistics</h3>
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-label">Current Epoch</div>
                    <div class="stat-value">{latest_epoch}/50</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Progress</div>
                    <div class="stat-value">{latest_epoch/50*100:.0f}%</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Best mAP@50</div>
                    <div class="stat-value">{max(data['map50']):.3f}</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Improvement</div>
                    <div class="stat-value">{(max(data['map50'])-data['map50'][0])/data['map50'][0]*100:+.1f}%</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const epochs = {json.dumps(data['epochs'])};
        const map50 = {json.dumps([x*100 for x in data['map50']])};
        const map95 = {json.dumps([x*100 for x in data['map95']])};
        const precision = {json.dumps([x*100 for x in data['precision']])};
        const recall = {json.dumps([x*100 for x in data['recall']])};
        
        // mAP Chart
        new Chart(document.getElementById('mapChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [
                    {{
                        label: 'mAP@50',
                        data: map50,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 3,
                        pointBackgroundColor: '#667eea'
                    }},
                    {{
                        label: 'mAP@50-95',
                        data: map95,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 3,
                        pointBackgroundColor: '#f59e0b'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: true, position: 'top' }} }},
                scales: {{
                    y: {{ beginAtZero: true, max: 100 }}
                }}
            }}
        }});
        
        // Precision & Recall Chart
        new Chart(document.getElementById('prChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [
                    {{
                        label: 'Precision',
                        data: precision,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 3,
                        pointBackgroundColor: '#10b981'
                    }},
                    {{
                        label: 'Recall',
                        data: recall,
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        tension: 0.4,
                        fill: true,
                        pointRadius: 3,
                        pointBackgroundColor: '#ef4444'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: true, position: 'top' }} }},
                scales: {{
                    y: {{ beginAtZero: true, max: 100 }}
                }}
            }}
        }});
        
        // All Metrics Chart
        new Chart(document.getElementById('allChart'), {{
            type: 'line',
            data: {{
                labels: epochs,
                datasets: [
                    {{
                        label: 'mAP@50',
                        data: map50,
                        borderColor: '#667eea',
                        pointRadius: 2
                    }},
                    {{
                        label: 'mAP@50-95',
                        data: map95,
                        borderColor: '#f59e0b',
                        pointRadius: 2
                    }},
                    {{
                        label: 'Precision',
                        data: precision,
                        borderColor: '#10b981',
                        pointRadius: 2
                    }},
                    {{
                        label: 'Recall',
                        data: recall,
                        borderColor: '#ef4444',
                        pointRadius: 2
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: true, position: 'top' }} }},
                scales: {{
                    y: {{ beginAtZero: true, max: 100 }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

with open(r'd:\Github\EV-PassengerDection-RL\results\coco_full\passenger_detection5\training_report.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("✅ Interactive training report generated!")
print("📍 Open: results/coco_full/passenger_detection5/training_report.html")
